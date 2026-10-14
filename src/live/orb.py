r"""
ORB SHADOW ENGINE (9 Oct 2026) - NAKLI trades, KOI ORDER NAHI. Zone bot ke saath chalta hai, apna alag virtual account.

Strategy (backtest: src/research/orb_nr_test.py + orb_ml_test.py, model: src/research/orb_train.py):
  Dabbe (NY time): ASIA 19:00-02:00, LON30 03:00-03:30, LON60 03:00-04:00. Breakout 11:00 tak, trade 16:00 tak.
  1. Dabba bana (range end): high/low = bid M5 candles. COMPRESSION check: NR4 (kal ki daily range 4 din mein
     sabse chhoti) YA TIGHT (dabba / pichle 20 dabbon ka average <= 30th pct). Nahi -> aaj is dabbe pe trade nahi.
  2. 11:00 tak pehla tick jo dabba tode (BUY: bid > high, fill ask par; SELL: bid < low, fill bid par).
     Sirf PEHLA break gina jaata hai.
  3. Break pe ML (models/orb_brain.txt) score; score >= threshold (top 30%) -> trade, warna skip.
  4. Ek pair pe ek din max 1 ORB trade (pehle wala jeetta hai).
  5. Trade: SL = dabbe ki doosri side, TP = 2R, 16:00 NY tak band nahi hua toh wahin close. Har tick pe
     (src/backtest/tick_engine.py, wahi rules jo backtest mein).
  Account: alag $10,000, har trade 0.4% risk (9 Oct: 0.25 -> 0.4 on his ask; backtest 0.4% = ~5.6%/yr, maxDD ~10%), $5/lot commission + 0.2 pip stop slippage (backtest jaisa).
Files (data/live): orb_state.json (dashboard + restart), orb_trades.csv (band trades), orb_decisions.csv (har dabba).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr, structure_breaks  # noqa: E402
from src.live.live_features import frames  # noqa: E402
from src.risk.portfolio_sim import LOT, PIP, risk_usd_per_unit  # noqa: E402

LIVE = ROOT / "data" / "live"
NY = "America/New_York"
MODEL = ROOT / "models" / "orb_brain.txt"
META = ROOT / "models" / "orb_meta.json"
RANGES = {"ASIA": ((19, 0), (2, 0), (11, 0)), "LON30": ((3, 0), (3, 30), (11, 0)), "LON60": ((3, 0), (4, 0), (11, 0))}
EOD = (16, 0)
SLIP_PIPS = 0.2
FINAL = {"skip_nocomp", "no_break", "skip_ml", "skip_daily", "skip_lot", "traded", "no_data"}
DEC_COLS = ["day", "symbol", "range", "status", "reason", "hi", "lo", "w", "w_ratio", "nr4", "tight", "direction",
            "break_ms", "entry", "sl", "score", "threshold"]
TRADE_COLS = ["day", "symbol", "range", "direction", "fill_time", "exit_time", "entry", "sl", "tp", "score", "lots",
              "risk_usd", "r", "exit_reason", "pnl", "equity"]


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def windows(day, rng: str) -> tuple[int, int, int, int]:
    """(range start, range end, break window end, EOD) as UTC ms for NY trading date `day`."""
    (h0, m0), (h1, m1), (h2, m2) = RANGES[rng]
    d = pd.Timestamp(day).tz_localize(NY) if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day)
    d0 = d.normalize()
    rs = (d0 - pd.Timedelta(days=1) if h0 > h1 else d0) + pd.Timedelta(hours=h0, minutes=m0)
    # DST-safe: build wall-clock times in NY
    rs = pd.Timestamp(rs.tz_localize(None)).tz_localize(NY)
    mk = lambda h, m: pd.Timestamp(d0.tz_localize(None) + pd.Timedelta(hours=h, minutes=m)).tz_localize(NY)  # noqa: E731
    return _ms(rs.tz_convert("UTC")), _ms(mk(h1, m1).tz_convert("UTC")), _ms(mk(h2, m2).tz_convert("UTC")), \
        _ms(mk(*EOD).tz_convert("UTC"))


def bar_hilo(m5: pd.DataFrame, t0: int, t1: int):
    """Bid high / low of closed M5 bars with t0 <= open < t1 (same as bid ticks max / min)."""
    ix = m5.index.as_unit("ns").asi8 // 1_000_000
    sel = (ix >= t0) & (ix < t1)
    if sel.sum() < 3:
        return None
    return float(m5["high"].to_numpy()[sel].max()), float(m5["low"].to_numpy()[sel].min())


class OrbBrain:
    def __init__(self):
        import lightgbm as lgb
        self.meta = json.loads(META.read_text())
        self.model = lgb.Booster(model_file=str(MODEL))
        self.threshold = float(self.meta["threshold"])
        self.features = self.meta["features"]

    def score(self, f: dict) -> float:
        X = pd.DataFrame([{k: f.get(k, np.nan) for k in self.features}])
        X["symbol"] = pd.Categorical(X["symbol"], categories=self.meta["symbols"])
        X["range"] = pd.Categorical(X["range"], categories=self.meta["ranges"])
        for c in self.features:
            if c not in ("symbol", "range"):
                X[c] = X[c].astype(float)
        return float(self.model.predict(X)[0])


def range_features(symbol: str, rng: str, day, m5: pd.DataFrame, hi: float, lo: float, r_end: int, q30: dict) -> dict:
    """Compression features known at range end: nr3..nr7, inside, w_ratio, tight + D1 context for later."""
    w = hi - lo
    # previous 20 same ranges (weekdays before `day`, from M5 bars)
    widths = []
    dd = pd.Timestamp(day)
    k = 1
    while len(widths) < 20 and k <= 40:
        p = (dd - pd.Timedelta(days=k)).date()
        k += 1
        if pd.Timestamp(p).weekday() >= 5:
            continue
        a, b, _, _ = windows(p, rng)
        hl = bar_hilo(m5, a, b)
        if hl is not None and hl[0] > hl[1]:
            widths.append(hl[0] - hl[1])
    w_ratio = w / float(np.mean(widths)) if len(widths) >= 10 else np.nan
    now = pd.Timestamp(r_end, unit="ms", tz="UTC")
    d1 = frames(m5[m5.index < now], now)["D1"].copy()
    out = dict(w=w, w_ratio=w_ratio, tight=int(bool(w_ratio <= q30[rng])) if w_ratio == w_ratio else 0)
    if len(d1) >= 9:
        rngs = (d1.high - d1.low).to_numpy()
        i = len(d1) - 1
        for n in range(3, 8):
            out[f"nr{n}"] = int(rngs[i] <= rngs[i - n + 1:i + 1].min())
        out["inside"] = int(d1.high.iloc[i] <= d1.high.iloc[i - 1] and d1.low.iloc[i] >= d1.low.iloc[i - 1])
        out["d1_atr"] = float(atr(d1).iloc[i])
        out["d1_o"], out["d1_h"], out["d1_l"], out["d1_c"] = (float(d1[c].iloc[i]) for c in ("open", "high", "low", "close"))
        out["d1_c20"] = float(d1.close.iloc[max(i - 20, 0)])
    else:
        out["nr4"] = 0
    return out


def break_features(rec: dict, m5: pd.DataFrame, break_ms: int, bull: bool, entry: float, risk: float,
                   tick_spread: float) -> dict:
    """ML features at the break tick (same formulas as src/research/orb_ml_test.features)."""
    sg = 1 if bull else -1
    f = {k: rec.get(k) for k in ("nr3", "nr4", "nr5", "nr6", "nr7", "inside", "w_ratio")}
    f.update(symbol=rec["symbol"], range=rec["range"])
    f["range_atr"] = rec["w"] / rec["d1_atr"] if rec.get("d1_atr") else np.nan
    f["d1_mom_aligned"] = int(np.sign(rec["d1_c"] - rec["d1_c20"]) == sg) if "d1_c" in rec else np.nan
    f["prevday_aligned"] = int(np.sign(rec["d1_c"] - rec["d1_o"]) == sg) if "d1_c" in rec else np.nan
    if "d1_h" in rec:
        pos = (entry - rec["d1_l"]) / max(rec["d1_h"] - rec["d1_l"], 1e-12)
        f["prevday_pos"] = pos if bull else 1 - pos
    if rec["range"] != "ASIA" and rec.get("a_hi") is not None and rec["a_hi"] > rec["a_lo"]:
        ap = (entry - rec["a_lo"]) / (rec["a_hi"] - rec["a_lo"])
        f["asia_pos"] = ap if bull else 1 - ap
    else:
        f["asia_pos"] = np.nan
    bt = pd.Timestamp(break_ms, unit="ms", tz="UTC")
    bar0 = bt.floor("5min")
    before = m5[m5.index < bar0]
    h4 = frames(before, bt)["H4"]
    br = structure_breaks(h4, 2)
    br = br[br.time <= bt]
    f["h4_aligned"] = int((br.direction.iloc[-1] == "bull") == bull) if len(br) else np.nan
    f["break_delay_min"] = (break_ms - rec["r_end"]) / 60000
    f["weekday"] = pd.Timestamp(rec["day"]).weekday()
    v = before["volume"].to_numpy()
    f["act_15m"] = float(v[-3:].sum() / 3 / max(v[-288:].sum() / 288, 1e-9)) if len(v) >= 288 else np.nan
    sp = float(m5.loc[bar0, "spread"]) if bar0 in m5.index else tick_spread    # replay: bar spread (= backtest)
    f["spread_r"] = sp / risk
    f["risk_pips"] = risk / PIP[rec["symbol"]]
    return f


class OrbEngine:
    def __init__(self, feed, symbols=None, out_dir: Path = LIVE, log=print, capital: float = 10_000.0,
                 risk_pct: float = 0.4, resume: bool = True, write_view: bool = True, news_guard=None):
        self.feed, self.log, self.out, self.write_view = feed, log, out_dir, write_view
        self.news = news_guard          # 15 Oct: NEWS BLOCK -5/+20 min (config.NEWS_BLOCK_*), same as zone bot
        self.exec = None                # 15 Oct: src/live/execution.Executor (real DEMO orders) - set by run_live
        self.symbols = symbols or config.CORE_SYMBOLS
        self.brain = OrbBrain()
        self.q30 = self.brain.meta["tight_q30"]
        self.capital, self.risk_pct = capital, risk_pct
        self.commission = config.BROKER_COMMISSION
        self.equity = self.peak = capital
        self.open: list[dict] = []
        self.recs: dict = {}            # "day|symbol|range" -> record
        self.traded: set = set()        # "day|symbol" with an ORB trade
        self.n_closed = 0
        self.close_done: set = set()    # 15 Oct: dashboard ORB CLOSE commands already done (control.json "orb_close")
        self._ctl_mtime = None
        self.keep_days = 3              # dashboard / restart ke liye kitne din ke dabbe yaad rakhne hain
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_file = self.out / "orb_state.json"
        self.trades_file = self.out / "orb_trades.csv"
        self.dec_file = self.out / "orb_decisions.csv"
        if resume and self.state_file.exists():
            self._load()
        self.log(f"ORB: brain trained {self.brain.meta['trained_at']}, threshold {self.brain.threshold:.3f} (top 30%), "
                 f"equity ${self.equity:,.2f}, open {len(self.open)}")

    # ------------------------------------------------------------------
    def _now_ms(self, now_ms):
        if now_ms is not None:
            return int(now_ms)
        clock = getattr(self.feed, "clock", None)
        return _ms(clock) if clock is not None else _ms(pd.Timestamp.now(tz="UTC"))

    def _days(self, now_ms: int):
        ny = pd.Timestamp(now_ms, unit="ms", tz="UTC").tz_convert(NY)
        out = []
        for k in (0, 1):
            d = (ny + pd.Timedelta(days=k)).date()
            if pd.Timestamp(d).weekday() < 5:
                out.append(d)
        return out

    def step(self, now_ms: int | None = None) -> bool:
        """Call every ~1 s (live) or every candle (replay). True = something changed (save)."""
        now = self._now_ms(now_ms)
        changed = False
        for day in self._days(now):
            for sym in self.symbols:
                for rng in RANGES:
                    key = f"{day}|{sym}|{rng}"
                    rec = self.recs.get(key)
                    rs, re_, we, eod = windows(day, rng)
                    if rec is None:
                        if now < rs - 3600_000:
                            continue
                        rec = self.recs[key] = dict(day=str(day), symbol=sym, range=rng, r_start=rs, r_end=re_,
                                                    w_end=we, eod=eod, status="waiting", reason="")
                        changed = True
                    if rec["status"] in FINAL or rec["status"] == "armed" and now < re_:
                        continue
                    try:
                        changed |= self._advance(rec, now)
                    except Exception as e:  # noqa: BLE001 - ORB kabhi bot na roke
                        self.log(f"ORB {key}: {e.__class__.__name__}: {e}")
        changed |= self._manage(now)
        changed |= self._manual_close(now)
        if changed:
            self._prune(now)
        return changed

    def _advance(self, rec: dict, now: int) -> bool:
        st = rec["status"]
        if now < rec["r_start"]:
            return False
        if now < rec["r_end"]:
            if st != "forming":
                rec["status"], rec["reason"] = "forming", "range is forming"
                return True
            return False
        if st in ("waiting", "forming"):
            return self._build(rec, now)
        if st == "armed":
            return self._watch_break(rec, now)
        return False

    def _build(self, rec: dict, now: int) -> bool:
        m5, _ = self.feed.bars(rec["symbol"])
        last_open = _ms(m5.index[-1]) if len(m5) else 0
        if last_open < rec["r_end"] - 300_000:                    # last range candle not closed yet
            if now > rec["r_end"] + 15 * 60_000:
                self._final(rec, "no_data", "no candles for the range")
                return True
            return False
        hl = bar_hilo(m5, rec["r_start"], rec["r_end"])
        if hl is None or hl[0] <= hl[1]:
            self._final(rec, "no_data", "not enough candles in the range")
            return True
        rec["hi"], rec["lo"] = hl
        rec.update(range_features(rec["symbol"], rec["range"], rec["day"], m5, hl[0], hl[1], rec["r_end"], self.q30))
        if rec["range"] != "ASIA":
            a0, a1, _, _ = windows(rec["day"], "ASIA")
            ahl = bar_hilo(m5, a0, a1)
            rec["a_hi"], rec["a_lo"] = (ahl if ahl else (None, None))
        comp = rec.get("nr4") == 1 or rec.get("tight") == 1
        if not comp:
            self._final(rec, "skip_nocomp", f"no compression (NR4 {rec.get('nr4')}, range {rec['w_ratio']:.2f}x normal, "
                                            f"tight <= {self.q30[rec['range']]:.2f}x)" if rec.get("w_ratio") == rec.get("w_ratio")
                        else "no compression")
            return True
        why = []
        if rec.get("nr4") == 1:
            why.append("NR4")
        if rec.get("tight") == 1:
            why.append(f"tight {rec['w_ratio']:.2f}x")
        rec["status"], rec["reason"] = "armed", "compression " + " + ".join(why) + " -> buy stop / sell stop ready"
        rec["last_ms"] = rec["r_end"] - 1
        self.log(f"ORB {rec['symbol']} {rec['range']} {rec['day']}: range {rec['lo']}-{rec['hi']} ARMED ({', '.join(why)})")
        return True

    def _watch_break(self, rec: dict, now: int) -> bool:
        to = min(now, rec["w_end"])
        if to > rec["last_ms"] + 1:
            tk = self.feed.ticks(rec["symbol"], rec["last_ms"] + 1, to)
            if tk is not None and len(tk[0]):
                t, a, b = tk
                hit = np.nonzero((b > rec["hi"]) | (b < rec["lo"]))[0]
                if len(hit):
                    i = int(hit[0])
                    return self._on_break(rec, int(t[i]), float(a[i]), float(b[i]))
                rec["last_ms"] = int(t[-1])
            elif tk is not None:
                pass
        if now >= rec["w_end"]:
            self._final(rec, "no_break", "range not broken before 11:00 NY")
            return True
        return False

    def _on_break(self, rec: dict, ms: int, ask: float, bid: float) -> bool:
        bull = bid > rec["hi"]
        entry = ask if bull else bid
        sl = rec["lo"] if bull else rec["hi"]
        risk = (entry - sl) if bull else (sl - entry)
        rec.update(direction="bull" if bull else "bear", break_ms=ms, entry=entry, sl=sl)
        if risk <= 0:
            self._final(rec, "no_data", "bad risk")
            return True
        m5, _ = self.feed.bars(rec["symbol"])
        f = break_features(rec, m5, ms, bull, entry, risk, ask - bid)
        s = self.brain.score(f)
        rec["score"], rec["features"] = round(s, 4), {k: (None if (isinstance(v, float) and v != v) else v) for k, v in f.items()}
        side = "BUY" if bull else "SELL"
        if s < self.brain.threshold:
            self._final(rec, "skip_ml", f"{side} break, ML {s:.3f} < {self.brain.threshold:.3f}")
            return True
        dk = f"{rec['day']}|{rec['symbol']}"
        if dk in self.traded:
            self._final(rec, "skip_daily", f"{side} break, ML {s:.3f} ok but already 1 ORB trade on this pair today")
            return True
        hit = self._news_hit(rec["symbol"], ms)
        if hit:
            self._final(rec, "skip_news", f"{side} break, ML {s:.3f} ok but inside news block: {hit}")
            self.log(f"ORB {rec['symbol']} {rec['range']} {side} break skipped - news block ({hit})")
            return True
        per_unit = risk_usd_per_unit(rec["symbol"], entry, risk)
        lots = float(np.floor((self.equity * self.risk_pct / 100 / per_unit) / LOT / 0.01) * 0.01)
        if lots < 0.01:
            self._final(rec, "skip_lot", "lot below 0.01")
            return True
        tr = dict(kind="orb", day=rec["day"], symbol=rec["symbol"], range=rec["range"], direction=rec["direction"],
                  bull=bull, entry=entry, sl=sl, risk=risk, tp=entry + (2 if bull else -2) * risk, exit="tp2",
                  score=round(s, 4), lots=round(lots, 2), risk_usd=round(lots * LOT * per_unit, 2), end_ms=rec["eod"],
                  fill_time=pd.Timestamp(ms, unit="ms", tz="UTC").isoformat(timespec="seconds"))
        te.init_state(tr, ms)
        self.open.append(tr)
        if self.exec is not None:                      # real DEMO market order (if ORB execution is ON)
            try:
                self.exec.on_orb_open(tr)
            except Exception as e:  # noqa: BLE001 - real order problem must not stop the shadow
                self.log(f"ORB exec open error: {e.__class__.__name__}: {e}")
        self.traded.add(dk)
        self._final(rec, "traded", f"{side} {lots:.2f} lots @ {entry}, SL {sl}, TP {tr['tp']:.5f}, ML {s:.3f}")
        self.log(f"ORB TRADE {rec['symbol']} {rec['range']} {side} @ {entry} SL {sl} TP {tr['tp']:.5f} "
                 f"lots {lots:.2f} ML {s:.3f}")
        return True

    def _news_hit(self, symbol: str, ms: int) -> str | None:
        """Event name if the break time is inside a NEWS BLOCK window (-5/+20 min of a high-impact event of the pair's
        currencies), else None. orb_news_test.py (15 Oct): 17 of 1,417 trades 2019-26, PF 0.19 -> block = PF 1.16 -> 1.18."""
        if self.news is None or not getattr(self.news, "block_on", False):
            return None
        try:
            t = pd.Timestamp(ms, unit="ms", tz="UTC")
            b = self.news.block_windows(symbol)
            hit = b[(b.start <= t) & (t <= b.end)]
            return None if hit.empty else str(hit.event.iloc[0])
        except Exception as e:  # noqa: BLE001 - calendar problem must not stop ORB
            self.log(f"ORB news check failed: {e}")
            return None

    def _final(self, rec: dict, status: str, reason: str):
        rec["status"], rec["reason"] = status, reason
        row = {c: rec.get(c) for c in DEC_COLS}
        row["threshold"] = self.brain.threshold
        try:
            pd.DataFrame([row]).reindex(columns=DEC_COLS).to_csv(self.dec_file, mode="a", header=not self.dec_file.exists(),
                                                                 index=False)
        except Exception as e:  # noqa: BLE001
            self.log(f"ORB decision log: {e}")

    def _manage(self, now: int) -> bool:
        changed, still = False, []
        for t in self.open:
            to = min(now, t["end_ms"])
            res = None
            if to > t["last_ms"] + 1:
                tk = self.feed.ticks(t["symbol"], t["last_ms"] + 1, to)
                if tk is not None and len(tk[0]):
                    res = te.feed_ticks(t, *tk)
                    changed = True
            if res is None and now >= t["end_ms"]:
                res = (t["r_now"], "eod", t["last_ms"])
            if res is None:
                still.append(t)
                continue
            self._close(t, *res)
            changed = True
        self.open = still
        return changed

    def _manual_close(self, now: int) -> bool:
        """Dashboard CLOSE button on an ORB trade -> close at the last tick price (ticks were just fed in _manage)."""
        from src.live import control
        p = self.out / control.FILE
        try:
            mt = p.stat().st_mtime if p.exists() else None
        except OSError:
            return False
        if mt is None or (mt == self._ctl_mtime and not getattr(self, "_ctl_pending", False)):
            return False
        self._ctl_mtime = mt
        cmds = [x for x in control.load(self.out).get("orb_close", []) if x["id"] not in self.close_done]
        self._ctl_pending = False
        changed = False
        for x in cmds:
            t = next((t for t in self.open if t["symbol"] == x["symbol"]
                      and int(pd.Timestamp(t["fill_time"]).timestamp()) == int(x["fill_time"])), None)
            if t is None:
                self.close_done.add(x["id"])
                changed = True
                continue
            if now - int(t.get("last_ms", 0)) > 120_000:      # no fresh tick (market closed) -> try again later
                self._ctl_pending = True
                continue
            r = float(t.get("r_now", 0.0))
            self.open = [o for o in self.open if o is not t]
            self._close(t, r, "manual", int(t["last_ms"]))
            control.log_action(self.out, kind="orb_close", symbol=t["symbol"], direction=t["direction"],
                               fill_time=int(x["fill_time"]), from_=t.get("exit"), to="closed", r_now=round(r, 3),
                               score=t.get("score"), note=t.get("range"))
            self.close_done.add(x["id"])
            changed = True
        return changed

    def _close(self, t: dict, r: float, why: str, ms: int):
        sym = t["symbol"]
        if self.exec is not None:                      # real DEMO position: close too (if still open at the broker)
            try:
                self.exec.on_orb_close(t, r, why)
            except Exception as e:  # noqa: BLE001
                self.log(f"ORB exec close error: {e.__class__.__name__}: {e}")
        slip = 0.0
        if why == "sl":
            sp = SLIP_PIPS * PIP[sym]
            slip = t["lots"] * LOT * (sp if sym.endswith("USD") else sp / t["entry"])
        pnl = r * t["risk_usd"] - self.commission * t["lots"] - slip
        self.equity += pnl
        self.peak = max(self.peak, self.equity)
        self.n_closed += 1
        row = dict(t, r=round(float(r), 3), exit_reason=why, pnl=round(float(pnl), 2), equity=round(self.equity, 2),
                   exit_time=pd.Timestamp(ms, unit="ms", tz="UTC").isoformat(timespec="seconds"))
        pd.DataFrame([row]).reindex(columns=TRADE_COLS).to_csv(self.trades_file, mode="a",
                                                               header=not self.trades_file.exists(), index=False)
        self.log(f"ORB CLOSED {sym} {t['range']} {t['direction']} {why} R {r:+.2f} pnl ${pnl:+.2f} | ORB equity ${self.equity:,.2f}")

    def _prune(self, now: int):
        """Keep records of the last ~3 days only (dashboard + restart)."""
        cut = (pd.Timestamp(now, unit="ms", tz="UTC").tz_convert(NY) - pd.Timedelta(days=self.keep_days)).date()
        self.recs = {k: v for k, v in self.recs.items() if pd.Timestamp(v["day"]).date() >= cut}
        self.traded = {k for k in self.traded if pd.Timestamp(k.split("|")[0]).date() >= cut}

    # ------------------------------------------------------------------
    def save(self):
        st = dict(version=1, updated=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), capital=self.capital,
                  risk_pct=self.risk_pct, equity=round(self.equity, 2), peak=round(self.peak, 2), n_closed=self.n_closed,
                  threshold=self.brain.threshold, tight_q30=self.q30, open=self.open, recs=self.recs,
                  traded=sorted(self.traded), close_done=sorted(self.close_done)[-200:])
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, default=str))
        tmp.replace(self.state_file)

    def _load(self):
        try:
            st = json.loads(self.state_file.read_text())
        except Exception as e:  # noqa: BLE001
            self.log(f"ORB state read failed ({e}); fresh start")
            return
        self.equity, self.peak = float(st.get("equity", self.capital)), float(st.get("peak", self.capital))
        self.open, self.recs = st.get("open", []), st.get("recs", {})
        self.traded, self.n_closed = set(st.get("traded", [])), int(st.get("n_closed", 0))
        self.close_done = set(st.get("close_done", []))


def reset_state(out: Path = LIVE):
    """ORB ka hisaab fresh (purani files rename, delete nahi)."""
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    for f in ("orb_state.json", "orb_trades.csv", "orb_decisions.csv"):
        p = out / f
        if p.exists():
            p.replace(out / f"{p.stem}_old_{stamp}{p.suffix}")
