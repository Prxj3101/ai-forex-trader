r"""
SHADOW MODE: bot live market dekhta hai, decisions leta hai, NAKLI (virtual) trades track karta hai.
KOI ORDER NAHI BHEJTA.

Har nayi M5 candle pe (har pair):
  1. Pichli candle pe jo "armed" orders the: price entry tak aaya? -> shadow trade khula
  2. Khule shadow trades: backtest wale exit logic se update (SL / 1:1 / trailing / 3 din timeout)
  3. Zinda M15 zones -> features (live_features) -> ML score (brain) -> VA filter -> news rules -> risk rules
     -> TAKE (is candle ke liye limit order "armed") ya SKIP (reason ke saath)
  4. Log: data/live/decisions.csv (har zone ka decision badalne pe), data/live/shadow_trades.csv

STATE SAVE (restart pe yaad): data/live/shadow_state.json mein balance, peak, kill switch, is mahine / aaj ka
hisaab, khule trades aur what-if trades. Restart pe wahin se aage (band rehte hue jo trade SL/TP hua, MT5 ki
purani candles se pakad leta hai). Fresh shuru: python src\live\shadow.py --reset
Band hue trades: data/live/closed_trades.csv (append, kabhi overwrite nahi).

WHAT-IF LOG (data/live/whatif.csv): jis bhi zone ko price ne chhua, uski ek row - bot ne kya decide kiya
(taken / ML low / value area / quiet / news / limits ...) aur 1:1 aur trailing dono se kitna R milta (gross + net).
Isse demo ke baad pata chalega ki har filter aur exit brain live mein sahi kaam kar rahe hain ya nahi.

Virtual account: $10,000, risk 0.25% (+ house money: is mahine ke profit ka 5%, max 0.5%), brake 5% (risk aadha),
kill switch 9%, daily stop 1.5%, daily risk budget 2% (Funding Pips daily limit 3%),
max 3 trades, 1 per pair, max 2 same USD direction. (src/risk/portfolio_sim.RiskConfig)

Chalana:
    python src\live\shadow.py                         (MT5 demo, live; Ctrl+C se band)
    python src\live\shadow.py --replay 2026-09-21 2026-09-26   (purane data pe test, MT5 ki zarurat nahi)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.live.brain import Brain  # noqa: E402
from src.live.live_features import HISTORY_DAYS, candidates  # noqa: E402
from src.research.management import POLICIES, run_trade_full  # noqa: E402
from src.risk.portfolio_sim import (LOT, PIP, RiskConfig, daily_ok, risk_usd_per_unit,  # noqa: E402
                                    trade_risk_pct, trading_day, usd_side)

LIVE = ROOT / "data" / "live"
NY = "America/New_York"
POL = {p.name: p for p in POLICIES}
EXIT_POLICY = {"tp1": POL["tp1"], "trail": POL["trail1"]}
HOLD_BARS = 3 * 288
STATE_VERSION = 2
TS_KEYS = ("bar_time", "zone", "fill_time", "exit_time")
CLOSED_COLS = ["symbol", "direction", "zone", "fill_time", "exit_time", "entry", "sl", "sl_pips", "score", "va_pos", "act",
               "exit", "exit_reason", "r", "lots", "risk_pct", "risk_usd", "braked", "pnl"]


def _category(reason: str | None) -> str:
    """Skip reason -> short label for the what-if log."""
    if reason is None:
        return "taken"
    r = reason.lower()
    for key, lab in (("ml score", "ml_low"), ("value area", "value_area"), ("market quiet", "quiet"),
                     ("news", "news"), ("kill", "kill_switch"), ("daily", "daily_limit"), ("lot below", "lot_small")):
        if key in r:
            return lab
    return "limits"            # max open / already in pair / USD one way


def _to_json(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.Timestamp):
            v = v.isoformat()
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def _from_json(d: dict) -> dict:
    d = dict(d)
    for k in TS_KEYS:
        if d.get(k) is not None:
            d[k] = pd.Timestamp(d[k])
    return d


def _pips(symbol, v):
    """price units -> pips (log ke liye)."""
    if v is None or pd.isna(v):
        return None
    return round(float(v) / PIP[symbol], 2)


def _unix(t):
    try:
        return int(pd.Timestamp(t).timestamp())
    except Exception:  # noqa: BLE001
        return None


def _num(v, nd=5):
    try:
        v = float(v)
        return None if np.isnan(v) else round(v, nd)
    except Exception:  # noqa: BLE001
        return None


def _watch_row(r, o, reason) -> dict:
    """Ek zone jo bot abhi dekh raha hai -> dashboard watchlist / chart box."""
    return dict(direction=o["direction"], entry=o["entry"], sl=o["sl"], sl_pips=o["sl_pips"], score=o["score"],
                exit=o["exit"], lots=o["lots"], status="ready" if reason is None else _category(reason),
                reason=reason or "", zone_candle_time=_unix(r.zone_candle_time), created_at=_unix(r.created_at),
                top=_num(getattr(r, "top", None)), bottom=_num(getattr(r, "bottom", None)),
                body_top=_num(getattr(r, "body_top", None)), body_bottom=_num(getattr(r, "body_bottom", None)),
                disp_atr=_num(getattr(r, "disp_atr", None), 2), has_fvg=bool(getattr(r, "has_fvg", False)))


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    tmp.replace(path)


def wf_commission():
    from src.ml import walkforward as wf
    return wf.COMMISSION


class ShadowEngine:
    def __init__(self, feed, symbols=None, capital: float = 10_000.0, out_dir: Path = LIVE, log=print,
                 news_guard=None, resume: bool = False, spread_floor: bool | None = None, write_view: bool = True):
        self.feed, self.log = feed, log
        self.write_view = write_view        # dashboard ke liye watchlist + chart candles likho (replay mein off)
        self.watch: dict = {}               # symbol -> zones jo bot abhi dekh raha hai (is candle ke candidates)
        self.spread_floor = config.ML_SPREAD_FLOOR if spread_floor is None else spread_floor
        self.symbols = symbols or config.CORE_SYMBOLS
        self.brain = Brain()
        self.cfg = RiskConfig(capital=capital, commission_per_lot=config.BROKER_COMMISSION)   # P&L = asli account
        self.equity = self.peak = capital
        self.day_pnl: dict = {}
        self.day_start: dict = {}           # broker trading day (NY 5 PM se) ke shuru ka balance
        self.month, self.month_start = None, capital   # house money: is mahine ka shuru ka balance
        self.open: list[dict] = []          # khule shadow trades
        self.closed: list[dict] = []
        self.armed: dict = {}               # symbol -> list of orders armed for the CURRENT candle
        self.last_decision: dict = {}       # zone key -> last decision text (sirf badlav log karte hain)
        self.kill = False
        self.wi_armed: dict = {}            # what-if: symbol -> every candidate of the CURRENT candle (taken or not)
        self.wi_open: list[dict] = []       # what-if: touched zones, tracked with BOTH exits
        self.filled_now: set = set()        # (symbol, zone, bar_time) that really filled
        self.news = news_guard
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.dec_file = self.out / "decisions.csv"
        if self.dec_file.exists():                   # purani file mein "act" column nahi -> alag rakh do
            head = self.dec_file.open().readline()
            if "spread_used" not in head.strip().split(","):   # purane columns -> alag rakh do
                stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                self.dec_file.replace(self.out / f"decisions_old_{stamp}.csv")
        self.log(f"shadow: brain trained {self.brain.trained_at}, top-30% threshold {self.brain.threshold:.3f}, "
                 f"profile {config.ACCOUNT_PROFILE}, VA filter {'ON' if config.VA_FILTER else 'OFF'}, "
                 f"activity filter {'ON (min ' + str(config.ACT_MIN) + ')' if config.ACT_FILTER else 'OFF'}, "
                 f"ML spread floor {'ON' if self.spread_floor else 'OFF'}, broker commission ${self.cfg.commission_per_lot}/lot")
        self.closed_file = self.out / "closed_trades.csv"
        self.wi_file = self.out / "whatif.csv"
        self.state_file = self.out / "shadow_state.json"
        if resume:
            self._load_state()
        c_ = self.cfg
        self.log(f"risk: {c_.risk_pct}% + house money {c_.house_money_pct}% of month profit (max {c_.risk_cap_pct}%), "
                 f"brake {c_.brake_dd_pct}% DD, kill {c_.kill_dd_pct}% DD, daily stop {c_.daily_loss_pct}%, "
                 f"daily risk budget {c_.daily_risk_budget_pct}%")

    def _day(self, t):
        d = trading_day(t)
        self.day_start.setdefault(d, self.equity)
        return d

    def _open_risk(self):
        return sum(t["risk_usd"] + self.cfg.commission_per_lot * t["lots"] for t in self.open)

    # ------------------------------------------------------------------
    def step(self, symbol: str):
        m5, bar_time = self.feed.bars(symbol)
        self._resolve_armed(symbol, m5, bar_time)
        self._resolve_whatif(symbol, m5, bar_time)
        self._update_open(symbol, m5)
        self._update_whatif(symbol, m5)
        self._decide(symbol, m5, bar_time)
        if self.write_view:
            self._write_view(symbol, m5, bar_time)

    # dashboard: chart candles + watchlist (sirf dikhane ke liye, trading pe asar nahi)
    def _write_view(self, symbol, m5, bar_time):
        try:
            tail = m5.tail(1500)
            ts = [int(t.timestamp()) for t in tail.index]          # unix seconds (index resolution se independent)
            bars = [[int(t), float(o), float(h), float(lw), float(c_)] for t, o, h, lw, c_ in
                    zip(ts, tail["open"], tail["high"], tail["low"], tail["close"])]
            _atomic_json(self.out / f"chart_{symbol}.json", {"symbol": symbol, "bar_time": int(bar_time.timestamp()),
                                                             "bars": bars})
            _atomic_json(self.out / "watchlist.json", {"updated": pd.Timestamp.now(tz="UTC").isoformat(),
                                                       "threshold": round(float(self.brain.threshold), 3),
                                                       "symbols": self.watch})
        except Exception as e:  # noqa: BLE001 - dashboard file kabhi bot na roke
            self.log(f"view: dashboard file likhne mein dikkat ({e.__class__.__name__}: {e})")

    # 1) armed orders of the previous candle: filled?
    def _resolve_armed(self, symbol, m5, bar_time):
        keep = []
        for o in self.armed.get(symbol, []):
            if o["bar_time"] >= bar_time:            # abhi wali candle chal rahi hai
                keep.append(o)
                continue
            if o["bar_time"] not in m5.index:
                continue
            b = m5.loc[o["bar_time"]]
            filled = (b["low"] + b["spread"] <= o["entry"]) if o["bull"] else (b["high"] >= o["entry"])
            if filled and (len(self.open) >= self.cfg.max_open or any(t["symbol"] == symbol for t in self.open)
                           or sum(t["usd_side"] == o["usd_side"] for t in self.open) >= self.cfg.max_usd_same_side):
                self._log_decision(symbol, o, "FILL-SKIPPED", "limit reached at fill (another order filled first)")
                continue
            if filled:
                d = self._day(o["bar_time"])
                why = daily_ok(self.cfg, self.day_start[d], self.day_pnl.get(d, 0.0), self._open_risk(),
                               o["risk_usd"] + self.cfg.commission_per_lot * o["lots"])
                if why:
                    self._log_decision(symbol, o, "FILL-SKIPPED", why)
                    continue
                t = dict(o, status="open", fill_time=o["bar_time"])
                self.open.append(t)
                self.filled_now.add((symbol, str(o["zone"]), o["bar_time"]))
                self._log_decision(symbol, o, "FILLED", f"lots {o['lots']}, exit {o['exit']}")
        self.armed[symbol] = keep

    # 2) open shadow trades: same exit logic as the backtest
    def _update_open(self, symbol, m5):
        hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
        idx = m5.index
        still = []
        for t in self.open:
            if t["symbol"] != symbol:
                still.append(t)
                continue
            f = int(idx.searchsorted(t["fill_time"]))
            g = min(f + HOLD_BARS, len(m5))
            r, why, k = run_trade_full(EXIT_POLICY[t["exit"]], t["bull"], t["entry"], t["risk"],
                                       hi[f:g], lo[f:g], cl[f:g], sp[f:g])
            done = why != "timeout" or (g - f) >= HOLD_BARS
            if not done:
                t["r_now"] = round(float(r), 3)
                still.append(t)
                continue
            slip = 0.0
            if why in ("sl", "trail", "be"):
                sp_ = self.cfg.slippage_pips * PIP[symbol]
                slip = t["lots"] * LOT * (sp_ if symbol.endswith("USD") else sp_ / t["entry"])
            pnl = r * t["risk_usd"] - self.cfg.commission_per_lot * t["lots"] - slip
            t.update(status="closed", r=round(float(r), 3), exit_reason=why, exit_time=idx[f + k] + pd.Timedelta(minutes=5),
                     pnl=round(float(pnl), 2))
            day = self._day(t["exit_time"])
            self.equity += pnl
            self.peak = max(self.peak, self.equity)
            self.day_pnl[day] = self.day_pnl.get(day, 0.0) + pnl
            self.closed.append(t)
            row = pd.DataFrame([_to_json(t)]).reindex(columns=CLOSED_COLS)      # fixed column order
            row.to_csv(self.closed_file, mode="a", header=not self.closed_file.exists(), index=False)
            self.log(f"  CLOSED {symbol} {t['direction']} {why} R {r:+.2f} pnl ${pnl:+.2f} | equity ${self.equity:,.2f}")
        self.open = still

    # 3) decisions for the current candle
    def _decide(self, symbol, m5, bar_time):
        c = candidates(symbol, m5, bar_time, spread_floor=self.spread_floor)
        self.armed.setdefault(symbol, [])
        self.watch[symbol] = {"bar_time": int(bar_time.timestamp()), "zones": []}
        if c.empty:
            return
        c = self.brain.score(c)
        dd = (self.peak - self.equity) / self.peak * 100
        if dd >= self.cfg.kill_dd_pct and not self.kill:
            self.kill = True
            self.log(f"!!! KILL SWITCH: drawdown {dd:.1f}% - naye trades band. Review karo; bot restart pe hi dobara chalega.")
        today = self._day(bar_time)
        mkey = bar_time.tz_convert(NY).strftime("%Y-%m")
        if mkey != self.month:
            self.month, self.month_start = mkey, self.equity
        risk_pct, braked = trade_risk_pct(self.cfg, self.equity, self.peak, self.month_start)
        wi = []
        for r in c.itertuples(index=False):
            o = self._order(symbol, r, bar_time, risk_pct, braked)
            reason = self._skip_reason(symbol, r, o, today, bar_time)
            wi.append(dict(o, label=_category(reason), reason=reason or "",
                           cost_r=float(r.cost_r) * self.cfg.commission_per_lot / wf_commission()
                           if pd.notna(r.cost_r) else None))
            self.watch[symbol]["zones"].append(_watch_row(r, o, reason))
            if reason is None:
                self.armed[symbol].append(o)
                self._log_decision(symbol, o, "TAKE", f"lots {o['lots']}, risk {o['risk_pct']:.2f}%, {o['exit']}")
            else:
                self._log_decision(symbol, o, "SKIP", reason)
        self.wi_armed[symbol] = wi

    def _order(self, symbol, r, bar_time, risk_pct, braked):
        bull = r.direction == "bull"
        per_unit = risk_usd_per_unit(symbol, r.entry, r.risk)
        lots = float(np.floor(((self.equity * risk_pct / 100) / per_unit) / LOT / self.cfg.lot_step) * self.cfg.lot_step)
        return dict(symbol=symbol, bar_time=bar_time, zone=r.zone_candle_time, direction=r.direction, bull=bull,
                    entry=float(r.entry), sl=float(r.sl), risk=float(r.risk), sl_pips=round(float(r.sl_pips), 1),
                    score=round(float(r.score), 3), ml_keep=bool(r.ml_keep), va_ok=bool(r.va_ok),
                    va_pos=round(float(r.r24_va_pos), 2) if pd.notna(r.r24_va_pos) else None,
                    act_ok=bool(r.act_ok), act=round(float(r.act_15m), 2) if pd.notna(r.act_15m) else None,
                    exit=r.exit_choice, lots=round(lots, 2), risk_usd=round(lots * LOT * per_unit, 2),
                    usd_side=usd_side(symbol, r.direction), braked=braked, risk_pct=round(risk_pct, 3),
                    spread_live=_pips(symbol, getattr(r, "spread_live", None)),
                    spread_used=_pips(symbol, getattr(r, "spread_used", None)))

    def _skip_reason(self, symbol, r, o, today, bar_time):
        if not o["ml_keep"]:
            return f"ML score {o['score']:.2f} < {self.brain.threshold:.2f}"
        if not o["va_ok"]:
            return f"price above value area ({o['va_pos']})"
        if not o["act_ok"]:
            return f"market quiet (activity {o['act']} < {config.ACT_MIN})"
        if self.kill:
            return "kill switch active"
        if self.news is not None and self.news.blocked(symbol, bar_time):
            return f"news window: {self.news.blocked(symbol, bar_time)}"
        why = daily_ok(self.cfg, self.day_start[today], self.day_pnl.get(today, 0.0), self._open_risk(),
                       o["risk_usd"] + self.cfg.commission_per_lot * o["lots"])
        if why:
            return why
        busy = self.open                        # backtest jaisa: sirf KHULE trades gine jaate hain
        if len(busy) >= self.cfg.max_open:
            return "max open trades"
        if any(b["symbol"] == symbol for b in busy):
            return "already in this pair"
        if sum(b["usd_side"] == o["usd_side"] for b in busy) >= self.cfg.max_usd_same_side:
            return "too much USD one way"
        if o["lots"] < self.cfg.min_lot:
            return "lot below 0.01"
        return None

    def _log_decision(self, symbol, o, decision, reason):
        key = (symbol, str(o["zone"]), o["direction"])
        text = f"{decision}|{reason.split(' ')[0] if decision == 'SKIP' else decision}"
        if decision in ("TAKE", "SKIP") and self.last_decision.get(key) == text:
            return                                   # same decision as last candle -> log mat karo
        self.last_decision[key] = text
        row = dict(logged_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), bar_time=o["bar_time"],
                   symbol=symbol, direction=o["direction"], zone=o["zone"], entry=o["entry"], sl=o["sl"],
                   sl_pips=o["sl_pips"], score=o["score"], va_pos=o["va_pos"], act=o["act"], exit=o["exit"], lots=o["lots"],
                   spread_live=o.get("spread_live"), spread_used=o.get("spread_used"),
                   decision=decision, reason=reason)
        pd.DataFrame([row]).to_csv(self.dec_file, mode="a", header=not self.dec_file.exists(), index=False)
        ny = pd.Timestamp(o["bar_time"]).tz_convert(NY).strftime("%m-%d %H:%M")
        self.log(f"  {ny} NY {symbol} {o['direction']:4s} zone@{o['entry']} ML {o['score']:.2f} -> {decision}: {reason}")

    # ------------------------------------------------------------------
    # WHAT-IF: har chhua hua zone, dono exits ke saath (taken ho ya skip)
    def _resolve_whatif(self, symbol, m5, bar_time):
        keep = []
        for o in self.wi_armed.get(symbol, []):
            if o["bar_time"] >= bar_time:
                keep.append(o)
                continue
            if o["bar_time"] not in m5.index:
                continue
            b = m5.loc[o["bar_time"]]
            if not ((b["low"] + b["spread"] <= o["entry"]) if o["bull"] else (b["high"] >= o["entry"])):
                continue
            label = o["label"]
            if (symbol, str(o["zone"]), o["bar_time"]) in self.filled_now:
                label = "taken"
            elif label == "taken":
                label = "fill_skipped"           # TAKE tha, par fill ke waqt limit / daily rule ne roka
            self.wi_open.append(dict(o, label=label, fill_time=o["bar_time"]))
        self.wi_armed[symbol] = keep
        self.filled_now = {k for k in self.filled_now if k[2] >= bar_time - pd.Timedelta(minutes=10)}

    def _update_whatif(self, symbol, m5):
        hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
        idx = m5.index
        still = []
        for w in self.wi_open:
            if w["symbol"] != symbol:
                still.append(w)
                continue
            f = int(idx.searchsorted(w["fill_time"]))
            g = min(f + HOLD_BARS, len(m5))
            res = {}
            for name in ("tp1", "trail"):
                r, why, k = run_trade_full(EXIT_POLICY[name], w["bull"], w["entry"], w["risk"],
                                           hi[f:g], lo[f:g], cl[f:g], sp[f:g])
                res[name] = (float(r), why, idx[min(f + k, len(idx) - 1)] + pd.Timedelta(minutes=5))
            if any(why == "timeout" for _, why, _ in res.values()) and (g - f) < HOLD_BARS:
                still.append(w)                   # koi exit abhi chal raha hai
                continue
            slip_r = self.cfg.slippage_pips * PIP[symbol] / w["risk"]
            cost = w.get("cost_r") or 0.0
            net = {n: r - cost - (slip_r if why in ("sl", "trail", "be") else 0.0) for n, (r, why, _) in res.items()}
            row = dict(logged_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), symbol=symbol,
                       direction=w["direction"], zone=w["zone"], fill_time=w["fill_time"], entry=w["entry"],
                       sl=w["sl"], sl_pips=w["sl_pips"], score=w["score"], va_pos=w["va_pos"], act=w["act"],
                       decision=w["label"], reason=w["reason"], exit_brain=w["exit"],
                       r_tp1=round(res["tp1"][0], 3), exit_tp1=res["tp1"][1], r_trail=round(res["trail"][0], 3),
                       exit_trail=res["trail"][1], trail_exit_time=res["trail"][2], cost_r=round(cost, 4),
                       net_tp1=round(net["tp1"], 3), net_trail=round(net["trail"], 3),
                       net_brain=round(net[w["exit"]], 3))
            pd.DataFrame([row]).to_csv(self.wi_file, mode="a", header=not self.wi_file.exists(), index=False)
        self.wi_open = still

    # ------------------------------------------------------------------
    # STATE: save / load (restart pe yaad rakhna)
    def _load_state(self):
        if not self.state_file.exists():
            self.log("state: koi purana hisaab nahi mila -> $%s se fresh shuru" % f"{self.cfg.capital:,.0f}")
            return
        st = json.loads(self.state_file.read_text())
        if st.get("version") != STATE_VERSION:
            self.log("state: purani file (bina full hisaab) -> fresh shuru")
            old = self.out / "shadow_trades.csv"
            if old.exists() and not self.closed_file.exists():
                old.replace(self.out / f"shadow_trades_before_state_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
            return
        self.equity, self.peak, self.kill = float(st["equity"]), float(st["peak"]), bool(st["kill"])
        self.month, self.month_start = st.get("month"), float(st.get("month_start", self.equity))
        from datetime import date
        self.day_start = {date.fromisoformat(k): float(v) for k, v in st.get("day_start", {}).items()}
        self.day_pnl = {date.fromisoformat(k): float(v) for k, v in st.get("day_pnl", {}).items()}
        self.open = [_from_json(t) for t in st.get("open_trades", [])]
        self.wi_open = [_from_json(w) for w in st.get("whatif_open", [])]
        self.n_closed_before = int(st.get("closed", 0))
        self.log(f"state: purana hisaab load (saved {st.get('updated')}): equity ${self.equity:,.2f}, peak ${self.peak:,.2f}, "
                 f"{len(self.open)} khule trade, {len(self.wi_open)} what-if, kill switch {'ON' if self.kill else 'off'}. "
                 f"Band rehte hue jo hua woh pehli candle pe update hoga.")

    def save(self):
        hist = pd.read_csv(self.closed_file) if self.closed_file.exists() else pd.DataFrame()
        rows = [dict(_to_json(t), r=t.get("r_now")) for t in self.open]
        snap = pd.concat([hist.assign(status="closed"),
                          pd.DataFrame(rows).reindex(columns=CLOSED_COLS).assign(status="open")], ignore_index=True) \
            if rows else hist.assign(status="closed")
        if len(snap):
            snap.to_csv(self.out / "shadow_trades.csv", index=False)
        pnl = hist["pnl"] if "pnl" in hist else pd.Series(dtype=float)
        state = dict(version=STATE_VERSION, equity=round(self.equity, 2), peak=round(self.peak, 2), kill=self.kill,
                     month=self.month, month_start=round(self.month_start, 2),
                     day_start={str(k): round(v, 2) for k, v in self.day_start.items()},
                     day_pnl={str(k): round(v, 2) for k, v in self.day_pnl.items()},
                     open_trades=[_to_json(t) for t in self.open],
                     whatif_open=[_to_json(w) for w in self.wi_open],
                     open=len(self.open), closed=int(len(hist)),
                     win_pct=round(100 * float((pnl > 0).mean()), 1) if len(pnl) else None,
                     risk=asdict(self.cfg), updated=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str))
        tmp.replace(self.state_file)               # atomic: crash ke beech file kharab nahi hoti


def reset_state(out: Path = LIVE):
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    for f in ("shadow_state.json", "closed_trades.csv", "shadow_trades.csv", "whatif.csv"):
        p = out / f
        if p.exists():
            p.replace(out / f"{p.stem}_old_{stamp}{p.suffix}")
    print(f"reset: purane files '_old_{stamp}' naam se rakh diye. Ab $10,000 se fresh shuru.")


def run_live(poll_s: int = 5):
    from src.live import monitor
    from src.live.feeds import MT5Feed
    monitor.start_log(LIVE)                      # bot window ka sab kuch data/live/bot.log mein bhi (dashboard ke liye)
    started_at = pd.Timestamp.now(tz="UTC")
    feed = MT5Feed(HISTORY_DAYS)
    guard = None
    if config.profile().get("news_rules") or config.NEWS_BLOCK:
        from src.data.news_live import live_events
        from src.risk.news_rules import NewsGuard
        guard = NewsGuard(live_events(), config.profile())
        print(f"news: {len(guard.events)} high-impact events this week, block {config.NEWS_BLOCK_BEFORE_MIN} min before "
              f"to {config.NEWS_BLOCK_AFTER_MIN} min after" if config.NEWS_BLOCK else "news: prop rules only")
    eng = ShadowEngine(feed, news_guard=guard, resume=True)
    last = {}
    news_hour = pd.Timestamp.now(tz="UTC").floor("h")
    print("Shadow mode chal raha hai (koi order nahi). Band karne ke liye Ctrl+C.")
    try:
        while True:
            if guard is not None and pd.Timestamp.now(tz="UTC").floor("h") != news_hour:   # har ghante calendar refresh
                news_hour = pd.Timestamp.now(tz="UTC").floor("h")
                try:
                    guard.events = live_events()
                    guard._cache.clear()
                    guard._bcache.clear()
                except Exception as e:  # noqa: BLE001 - calendar fail ho toh purana hi chalne do
                    print(f"news calendar refresh failed: {e}")
            for s in eng.symbols:
                bar_time = feed.current_bar_time(s)       # sasta check: sirf aakhri candle ka time
                if last.get(s) != bar_time:              # nayi M5 candle shuru hui
                    last[s] = bar_time
                    t0 = time.time()
                    eng.step(s)
                    print(f"{pd.Timestamp.now(tz=NY):%H:%M:%S} NY  {s} checked ({time.time() - t0:.1f}s) | "
                          f"equity ${eng.equity:,.2f} | open {len(eng.open)} | armed {len(eng.armed.get(s, []))}")
                    eng.save()
            monitor.write_health(feed, eng, LIVE, started_at, guard)   # dashboard: MT5 / prices / VPS sehat
            time.sleep(poll_s)
    except KeyboardInterrupt:
        eng.save()
        feed.close()
        print("Shadow mode band. Files: data/live/decisions.csv, shadow_trades.csv, shadow_state.json")


def run_replay(start: str, end: str, out: Path):
    from src.live.feeds import ReplayFeed
    feed = ReplayFeed(config.CORE_SYMBOLS, start, end, HISTORY_DAYS)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("decisions.csv", "shadow_trades.csv", "closed_trades.csv", "whatif.csv", "shadow_state.json"):
        (out / f).unlink(missing_ok=True)
    eng = ShadowEngine(feed, out_dir=out, spread_floor=False, write_view=False)   # Dukascopy = training data, floor nahi
    t0 = time.time()
    for i, _ in enumerate(feed):
        for s in eng.symbols:
            eng.step(s)
    eng.save()
    print(f"replay {start} -> {end}: {len(feed.times)} candles x {len(eng.symbols)} pairs in {time.time() - t0:.0f}s | "
          f"trades closed {len(eng.closed)}, open {len(eng.open)}, equity ${eng.equity:,.2f}")
    return eng


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shadow mode (no orders)")
    ap.add_argument("--replay", nargs=2, metavar=("START", "END"), help="purane data pe chalao, e.g. 2026-09-21 2026-09-26")
    ap.add_argument("--reset", action="store_true", help="purana hisaab hata ke $10,000 se fresh shuru")
    a = ap.parse_args()
    if a.reset and not a.replay:
        reset_state()
    if a.replay:
        run_replay(a.replay[0], a.replay[1], LIVE / "replay")
    else:
        run_live()
