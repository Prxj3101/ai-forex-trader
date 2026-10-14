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

Virtual account: $10,000, combo 0.5% / 0.25% (reduce at 3% DD, recover at 1% DD).
House money 5%, caps 1% / 0.5%; brake 5% (risk aadha), kill switch 10%.
Daily stop/budget: aggressive 2%/2.5%, conservative 1.5%/2%,
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
from src.live import control  # noqa: E402
from src.live.brain import Brain  # noqa: E402
from src.live.live_features import HISTORY_DAYS, candidates  # noqa: E402
from src.research.management import POLICIES, run_trade_full  # noqa: E402
from src.backtest import tick_engine as te  # noqa: E402
from src.risk.portfolio_sim import (LOT, PIP, daily_ok, risk_usd_per_unit,  # noqa: E402
                                    trade_risk_pct, trading_day, usd_side, RiskMode, combo_config)

LIVE = ROOT / "data" / "live"
NY = "America/New_York"
POL = {p.name: p for p in POLICIES}
EXIT_POLICY = {"tp1": POL["tp1"], "trail": POL["trail1"]}
HOLD_BARS = 3 * 288
STATE_VERSION = 2
TS_KEYS = ("bar_time", "zone", "fill_time", "exit_time")
WI_COLS = ["logged_at", "symbol", "direction", "zone", "fill_time", "entry", "sl", "sl_pips", "score", "va_pos", "act",
           "decision", "reason", "exit_brain", "r_tp1", "exit_tp1", "r_trail", "exit_trail", "trail_exit_time", "cost_r",
           "net_tp1", "net_trail", "net_brain", "user", "exit_bot", "net_bot"]
CLOSED_COLS = ["symbol", "direction", "zone", "fill_time", "exit_time", "entry", "sl", "sl_pips", "score", "va_pos", "act",
               "exit", "exit_reason", "r", "lots", "risk_pct", "risk_usd", "braked", "pnl"]


def _category(reason: str | None) -> str:
    """Skip reason -> short label for the what-if log."""
    if reason is None:
        return "taken"
    r = reason.lower()
    if r.startswith("taken manually"):
        return "taken_manual"
    for key, lab in (("avoided manually", "avoided"), ("ml score", "ml_low"), ("value area", "value_area"), ("market quiet", "quiet"),
                     ("news", "news"), ("kill", "kill_switch"), ("daily", "daily_limit"), ("lot below", "lot_small"),
                     ("already traded", "traded")):
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
                exit=o["exit"], exit_bot=o.get("exit_bot"), user_exit=o.get("user_exit", ""),
                taken=o.get("user") == "take", lots=o["lots"], status="ready" if reason is None else _category(reason),
                reason=reason or "", zone_candle_time=_unix(r.zone_candle_time), created_at=_unix(r.created_at),
                top=_num(getattr(r, "top", None)), bottom=_num(getattr(r, "bottom", None)),
                body_top=_num(getattr(r, "body_top", None)), body_bottom=_num(getattr(r, "body_bottom", None)),
                disp_atr=_num(getattr(r, "disp_atr", None), 2), has_fvg=bool(getattr(r, "has_fvg", False)))


def _live_levels(t: dict, hi, lo, sp) -> dict:
    """Khule trade ke chart levels: TP (sirf 1:1) aur abhi ka SL (trail mein +1R ke baad khiskta hai).
    management.run_trade_full jaisa hi: best = ab tak ka sabse accha R (band candles), trail = best - 1R."""
    bull, entry, risk = t["bull"], float(t["entry"]), float(t["risk"])
    if len(hi):
        fav = (hi - entry) / risk if bull else (entry - (lo + sp)) / risk
        best = max(0.0, float(np.max(fav)))
    else:
        best = 0.0
    pol = EXIT_POLICY[t["exit"]]
    stop_r = -1.0
    if pol.trail_r is not None and best >= pol.trail_start_r:
        stop_r = max(stop_r, best - pol.trail_r)
    sign = 1 if bull else -1
    return dict(best_r=round(best, 3),
                stop_now=round(entry + sign * stop_r * risk, 6),
                tp=round(entry + sign * pol.tp_r * risk, 6) if pol.tp_r is not None else None)


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
        self.traded_file = out_dir / "traded_zones.json"   # jin zones se trade hua -> chart pe box rehta hai
        try:
            self.traded: list = json.loads(self.traded_file.read_text()) if self.traded_file.exists() else []
        except Exception:  # noqa: BLE001
            self.traded = []
        self.spread_floor = config.ML_SPREAD_FLOOR if spread_floor is None else spread_floor
        self.symbols = symbols or config.CORE_SYMBOLS
        self.brain = Brain()
        self.cfg = combo_config(capital, config.BROKER_COMMISSION)
        self.risk_mode = RiskMode()
        self.equity = self.peak = capital
        self.day_pnl: dict = {}
        self.day_start: dict = {}           # broker trading day (NY 5 PM se) ke shuru ka balance
        self.month, self.month_start = None, capital   # house money: is mahine ka shuru ka balance
        self.open: list[dict] = []          # khule shadow trades
        self.closed: list[dict] = []
        self.armed: dict = {}               # symbol -> list of orders armed for the CURRENT candle
        self._last_bars: dict = {}          # symbol -> (closed M5 bars, current candle time) of the last step
        self._redecide: set = set()         # symbols to decide again after a mid-candle close
        self.last_decision: dict = {}       # zone key -> last decision text (sirf badlav log karte hain)
        self.kill = False
        self.wi_armed: dict = {}            # what-if: symbol -> every candidate of the CURRENT candle (taken or not)
        self.wi_open: list[dict] = []       # what-if: touched zones, tracked with BOTH exits
        self.filled_now: set = set()        # (symbol, zone, bar_time) that really filled
        self.news = news_guard
        self.out = out_dir
        # manual control (dashboard AVOID / CLOSE buttons)
        self.avoid: set = set()             # (symbol, zone unix, direction)
        self.parked: dict = {}              # symbol -> is candle ke orders jo avoid hue (UNDO pe wapas)
        self.control: dict = {k: [] for k in control.ZONE_LISTS + control.CMD_LISTS}
        self.take: set = set()              # (symbol, zone unix, direction) jo tune TAKE kiye (ML low ho toh bhi lo)
        self.exit_over: dict = {}           # (symbol, zone unix, direction) -> "tp1" / "trail" (tera exit, brain ki jagah)
        self.control_mtime = None
        self.close_done: set = set()        # close command ids jo ho chuke
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
        self.feat_file = self.out / "whatif_features.jsonl"   # har chhua zone: ML features + dono exits ka R + tera decision
        if self.wi_file.exists():                   # 15 Oct: what-if mein user / exit_bot columns aaye -> purani file alag
            head = self.wi_file.open().readline().strip().split(",")
            if "user" not in head:
                self.wi_file.replace(self.out / f"whatif_old_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
        self.state_file = self.out / "shadow_state.json"
        if resume:
            self._load_state()
        self.risk_mode.select(self.cfg, self.equity, self.peak)
        self.apply_control(None)            # tere avoid kiye zones startup se hi yaad
        c_ = self.cfg
        self.log(f"risk: {c_.risk_pct}% + house money {c_.house_money_pct}% of month profit (max {c_.risk_cap_pct}%), "
                 f"brake {c_.brake_dd_pct}% DD, kill {c_.kill_dd_pct}% DD, daily stop {c_.daily_loss_pct}%, "
                 f"daily risk budget {c_.daily_risk_budget_pct}%")
        self.log("risk combo: 0.5% -> 0.25% at 3% DD; back to 0.5% only at 1% DD")

    def _day(self, t):
        d = trading_day(t)
        self.day_start.setdefault(d, self.equity)
        return d

    def _open_risk(self):
        return sum(t["risk_usd"] + self.cfg.commission_per_lot * t["lots"] for t in self.open)

    # ------------------------------------------------------------------
    def step(self, symbol: str):
        m5, bar_time = self.feed.bars(symbol)
        self._last_bars[symbol] = (m5, bar_time)
        self.parked[symbol] = []                 # pichli candle ke avoided orders ab bekaar
        self._resolve_armed(symbol, m5, bar_time)
        self._resolve_whatif(symbol, m5, bar_time)
        self.manage_ticks(symbol, int(bar_time.value // 1_000_000), m5)   # SL / TP / trail on real ticks
        self._update_whatif(symbol, m5)
        self._decide(symbol, m5, bar_time)
        self._redecide.discard(symbol)
        if self.write_view:
            self._write_view(symbol, m5, bar_time)

    def redecide_pending(self) -> bool:
        """A trade closed mid-candle (tick SL/TP, manual close): limits like 'already in this pair' / max open /
        daily budget changed NOW -> decide again for the current candle (backtest also allows a new fill right
        after an exit). Refreshes the dashboard watchlist at once instead of after the next candle."""
        done = False
        for sym in list(self._redecide):
            if sym not in self._last_bars:
                continue
            m5, bt = self._last_bars[sym]
            self.armed[sym] = [o for o in self.armed.get(sym, []) if o["bar_time"] != bt]
            try:
                self._decide(sym, m5, bt)
                done = True
            except Exception as e:  # noqa: BLE001 - never stop the bot for this
                self.log(f"redecide {sym}: {e.__class__.__name__}: {e}")
            self._redecide.discard(sym)
        if done and self.write_view:
            self._write_watchlist()
        return done

    def _remember_traded_zone(self, t: dict):
        """Trade hua -> us zone ka box yaad rakho (watchlist se hat jaata hai, chart pe nahi hatna chahiye)."""
        if not self.write_view:
            return
        try:
            z = t.get("zbox") or {}
            self.traded.append(dict(z, symbol=t["symbol"], direction=t["direction"], fill_time=_unix(t["fill_time"]),
                                    entry=t["entry"], sl=t["sl"], exit=t["exit"]))
            self.traded = self.traded[-300:]
            _atomic_json(self.traded_file, self.traded)
        except Exception as e:  # noqa: BLE001
            self.log(f"view: traded zone save mein dikkat ({e.__class__.__name__}: {e})")

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
            if filled:
                # candle bolti hai bhara: asli ticks mein pehla touch dhoondo (fill se pehle ka move kabhi nahi ginna)
                b0 = int(o["bar_time"].value // 1_000_000)
                tk = self.feed.ticks(symbol, b0, b0 + 300_000) if hasattr(self.feed, "ticks") else None
                if tk is not None and len(tk[0]):
                    hit = np.nonzero(tk[1] <= o["entry"])[0] if o["bull"] else np.nonzero(tk[2] >= o["entry"])[0]
                    if not len(hit):
                        self._log_decision(symbol, o, "FILL-SKIPPED", "ticks never reached the entry (candle spread estimate)")
                        continue
                    self._fill(symbol, o, "candle", fill_ms=int(tk[0][hit[0]]))
                else:                                     # ticks nahi mile: candle ke END se manage (conservative)
                    self._fill(symbol, o, "candle", fill_ms=b0 + 300_000)
        self.armed[symbol] = keep

    def _fill(self, symbol, o, how: str, fill_ms: int | None = None) -> bool:
        """Armed limit order fill: risk limits dobara check -> khula shadow trade. how = 'tick' (live, turant) / 'candle'."""
        self.risk_mode.select(self.cfg, self.equity, self.peak)
        dd = (self.peak - self.equity) / self.peak * 100 if self.peak > 0 else 0
        if dd >= self.cfg.kill_dd_pct:
            self.kill = True
        if self.kill:
            self._log_decision(symbol, o, "FILL-SKIPPED", "kill switch active")
            return False
        if self.cfg.combo_enabled:
            mkey = o['bar_time'].tz_convert(NY).strftime('%Y-%m')
            if mkey != self.month:
                self.month, self.month_start = mkey, self.equity
            risk_pct, braked = trade_risk_pct(self.cfg, self.equity, self.peak, self.month_start)
            per_unit = risk_usd_per_unit(symbol, o['entry'], o['risk'])
            lots = float(np.floor((self.equity * risk_pct / 100 / per_unit) / LOT / self.cfg.lot_step) * self.cfg.lot_step)
            o = dict(o, lots=round(lots, 2), risk_usd=round(lots * LOT * per_unit, 2),
                     risk_pct=round(risk_pct, 3), braked=braked)
            if lots < self.cfg.min_lot:
                self._log_decision(symbol, o, "FILL-SKIPPED", "lot below 0.01")
                return False
        if (len(self.open) >= self.cfg.max_open or any(t["symbol"] == symbol for t in self.open)
                or sum(t["usd_side"] == o["usd_side"] for t in self.open) >= self.cfg.max_usd_same_side):
            self._log_decision(symbol, o, "FILL-SKIPPED", "limit reached at fill (another order filled first)")
            return False
        d = self._day(o["bar_time"])
        why = daily_ok(self.cfg, self.day_start[d], self.day_pnl.get(d, 0.0), self._open_risk(),
                       o["risk_usd"] + self.cfg.commission_per_lot * o["lots"])
        if why:
            self._log_decision(symbol, o, "FILL-SKIPPED", why)
            return False
        now = pd.Timestamp.now(tz="UTC")
        # fill_time = candle ka time (exit logic backtest jaisa usi candle se); filled_at = asli ghadi ka time
        t = dict(o, status="open", fill_time=o["bar_time"], r_now=0.0,
                 filled_at=(now if how == "tick" else o["bar_time"] + pd.Timedelta(minutes=5)).isoformat(timespec="seconds"))
        if fill_ms is None:
            fill_ms = int(now.value // 1_000_000)
        te.init_state(t, fill_ms)                           # tick-by-tick SL / TP / trail state (src/backtest/tick_engine.py)
        t["filled_at"] = pd.Timestamp(fill_ms, unit="ms", tz="UTC").isoformat(timespec="seconds")
        t.update(te.levels(t))                              # chart: TP + SL turant
        self.open.append(t)
        self._remember_traded_zone(t)
        self.filled_now.add((symbol, str(o["zone"]), o["bar_time"]))
        self._log_decision(symbol, o, "FILLED", f"lots {o['lots']}, exit {o['exit']}"
                           + (f" (live tick {now.tz_convert(NY):%H:%M:%S} NY)" if how == "tick" else ""))
        return True

    def fill_intrabar(self, prices: dict) -> bool:
        """Har ~1 s: live bid/ask ne armed limit chhua? -> turant fill (candle band hone ka 5 min wait nahi).
        BUY limit: ask <= entry, SELL limit: bid >= entry. Jo yahan miss ho (1 s ke beech ka chhota touch),
        woh candle band hone pe purane tareeke se (high/low) pakda jaata hai. True = koi trade khula."""
        changed = False
        for symbol, orders in self.armed.items():
            p = prices.get(symbol)
            if not orders or not p or p.get("tick_age_sec") is None or p["tick_age_sec"] > 60:
                continue
            keep = []
            for o in orders:
                hit = (p["ask"] <= o["entry"]) if o["bull"] else (p["bid"] >= o["entry"])
                if not hit:
                    keep.append(o)
                    continue
                ok = self._fill(symbol, o, "tick", fill_ms=p.get("time_ms"))   # fill ya FILL-SKIPPED - order armed se hat jaata hai
                self._mark_watch(symbol, o, "filled" if ok else "fill_skipped")
                changed = True
            self.armed[symbol] = keep
        return changed

    def _mark_watch(self, symbol, o, status):
        """Dashboard watchlist: is zone ko READY se FILLED karo (agli candle tak READY na dikhe)."""
        if not self.write_view:
            return
        try:
            zt = _unix(o["zone"])
            for z in self.watch.get(symbol, {}).get("zones", []):
                if z["zone_candle_time"] == zt and z["direction"] == o["direction"]:
                    z["status"], z["reason"] = status, "price reached the entry"
        except Exception as e:  # noqa: BLE001
            self.log(f"view: watchlist update mein dikkat ({e.__class__.__name__}: {e})")
        self._write_watchlist()

    # 2) open shadow trades: SL / TP / trailing on REAL TICKS (same rules as the tick backtest)
    def manage_ticks(self, symbol=None, upto_ms: int | None = None, m5=None) -> bool:
        """Every ~1 s (live) and every candle (replay): feed new ticks to each open trade. True = something closed.
        No ticks available -> old M5 method (with the fill-candle fix) for that pair."""
        upto_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000) if upto_ms is None else int(upto_ms)
        still, closed, m5_needed = [], False, set()
        for t in self.open:
            if symbol is not None and t["symbol"] != symbol:
                still.append(t)
                continue
            if "entry_ms" not in t:                     # trade from an older state file: manage from fill candle end
                te.init_state(t, int(pd.Timestamp(t["fill_time"]).value // 1_000_000) + 300_000)
            tk = self.feed.ticks(t["symbol"], t["last_ms"] + 1, upto_ms) if hasattr(self.feed, "ticks") else None
            if tk is None:
                m5_needed.add(t["symbol"])
                still.append(t)
                continue
            res = te.feed_ticks(t, *tk)
            if res is None and upto_ms >= t["entry_ms"] + te.HOLD_MS:
                res = (t["r_now"], "timeout", t["last_ms"])
            if res is None:
                t.update(te.levels(t))
                still.append(t)
                continue
            r, why, ms = res
            self._book_close(t, r, why, pd.Timestamp(ms, unit="ms", tz="UTC"))
            closed = True
        self.open = still
        for s_ in m5_needed:
            if m5 is not None and (symbol is None or s_ == symbol):
                self._update_open(s_, m5)
        return closed

    # fallback: M5 candles (only when ticks are not available)
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
                t.update(_live_levels(t, hi[f:g], lo[f:g], sp[f:g]))     # chart: TP + abhi ka (trailing) SL
                still.append(t)
                continue
            self._book_close(t, r, why, idx[f + k] + pd.Timedelta(minutes=5))
        self.open = still

    def _book_close(self, t, r, why, exit_time):
        """Band trade ka hisaab: P&L (R x risk - commission - stop slippage), equity, closed_trades.csv."""
        symbol = t["symbol"]
        slip = 0.0
        if why in ("sl", "trail", "be"):
            sp_ = self.cfg.slippage_pips * PIP[symbol]
            slip = t["lots"] * LOT * (sp_ if symbol.endswith("USD") else sp_ / t["entry"])
        pnl = r * t["risk_usd"] - self.cfg.commission_per_lot * t["lots"] - slip
        t.update(status="closed", r=round(float(r), 3), exit_reason=why, exit_time=exit_time, pnl=round(float(pnl), 2))
        day = self._day(t["exit_time"])
        self.equity += pnl
        self.peak = max(self.peak, self.equity)
        self.day_pnl[day] = self.day_pnl.get(day, 0.0) + pnl
        self.closed.append(t)
        self._redecide.update(self.symbols)          # limits changed -> watchlist / orders re-checked right away
        row = pd.DataFrame([_to_json(t)]).reindex(columns=CLOSED_COLS)      # fixed column order
        row.to_csv(self.closed_file, mode="a", header=not self.closed_file.exists(), index=False)
        self.log(f"  CLOSED {symbol} {t['direction']} {why} R {r:+.2f} pnl ${pnl:+.2f} | equity ${self.equity:,.2f}")

    # ------------------------------------------------------------------
    # MANUAL CONTROL (dashboard buttons): AVOID zone / CLOSE khula trade  (src/live/control.py)
    def _avoided(self, symbol, o) -> bool:
        return control.avoid_key(symbol, _unix(o["zone"]), o["direction"]) in self.avoid

    def apply_control(self, prices: dict | None) -> bool:
        """Har ~1 s: control.json badla? -> avoid list update + close commands. True = kuch badla (save karo)."""
        p = self.out / control.FILE
        try:
            mtime = p.stat().st_mtime if p.exists() else None
        except OSError:
            return False
        pending = [x for k in ("close", "trade_exit") for x in self.control.get(k, []) if x["id"] not in self.close_done]
        if mtime == self.control_mtime and not pending:
            return False
        if mtime != self.control_mtime:
            self.control_mtime = mtime
            self.control = control.load(self.out)
        changed = False
        new = {control.avoid_key(a["symbol"], a["zone"], a["direction"]) for a in self.control["avoid"]}
        if new != self.avoid:
            added, removed = new - self.avoid, self.avoid - new
            self.avoid = new
            for symbol in list(self.armed):
                keep = []
                for o in self.armed[symbol]:
                    if self._avoided(symbol, o):
                        self.parked.setdefault(symbol, []).append(o)      # isi candle mein UNDO ho toh wapas
                        self._log_decision(symbol, o, "SKIP", "avoided manually")
                    else:
                        keep.append(o)
                self.armed[symbol] = keep
            for symbol in list(self.parked):                              # UNDO: abhi wali candle ka order wapas armed
                back = [o for o in self.parked[symbol] if not self._avoided(symbol, o)]
                self.parked[symbol] = [o for o in self.parked[symbol] if self._avoided(symbol, o)]
                for o in back:
                    self.armed.setdefault(symbol, []).append(o)
                    self._log_decision(symbol, o, "TAKE", f"manual avoid removed, lots {o['lots']}, {o['exit']}")
            for k in added:
                self.log(f"  MANUAL: AVOID {k[0]} {k[2]} zone {pd.Timestamp(k[1], unit='s', tz='UTC').tz_convert(NY):%m-%d %H:%M} NY")
                control.log_action(self.out, kind="avoid", symbol=k[0], zone=k[1], direction=k[2], score=self._zone_score(k))
            for k in removed:
                self.log(f"  MANUAL: UNDO avoid {k[0]} {k[2]} zone {pd.Timestamp(k[1], unit='s', tz='UTC').tz_convert(NY):%m-%d %H:%M} NY")
                control.log_action(self.out, kind="unavoid", symbol=k[0], zone=k[1], direction=k[2])
            self._mark_watch_all()
            changed = True
        # TAKE (ML low ho toh bhi lo) + EXIT override (trade se pehle) -> us pair ka decision abhi dobara
        new_take = {control.zone_key(a["symbol"], a["zone"], a["direction"]) for a in self.control.get("take", [])}
        new_exit = {control.zone_key(a["symbol"], a["zone"], a["direction"]): a["exit"] for a in self.control.get("exit", [])}
        if new_take != self.take or new_exit != self.exit_over:
            for k in new_take - self.take:
                self.log(f"  MANUAL: TAKE {k[0]} {k[2]} zone {pd.Timestamp(k[1], unit='s', tz='UTC').tz_convert(NY):%m-%d %H:%M} NY")
                control.log_action(self.out, kind="take", symbol=k[0], zone=k[1], direction=k[2], score=self._zone_score(k))
            for k in self.take - new_take:
                control.log_action(self.out, kind="untake", symbol=k[0], zone=k[1], direction=k[2])
            for k in set(new_exit) | set(self.exit_over):
                if new_exit.get(k) != self.exit_over.get(k):
                    self.log(f"  MANUAL: EXIT {k[0]} {k[2]} zone -> {new_exit.get(k) or 'brain'}")
                    control.log_action(self.out, kind="exit_zone", symbol=k[0], zone=k[1], direction=k[2],
                                       from_=self.exit_over.get(k) or "brain", to=new_exit.get(k) or "brain")
            touched = {k[0] for k in (new_take ^ self.take)} | {k[0] for k in set(new_exit) ^ set(self.exit_over)} | \
                      {k[0] for k in new_exit if new_exit.get(k) != self.exit_over.get(k)}
            self.take, self.exit_over = new_take, new_exit
            self._redecide.update(touched)
            changed = True
        # TRADE_EXIT: khule trade ka exit beech mein badlo
        for x in [x for x in self.control.get("trade_exit", []) if x["id"] not in self.close_done]:
            t = next((t for t in self.open if t["symbol"] == x["symbol"] and _unix(t["fill_time"]) == int(x["fill_time"])), None)
            if t is None:
                self.close_done.add(x["id"])
                changed = True
                continue
            new, old = x["exit"], t["exit"]
            if new == old:
                self.close_done.add(x["id"])
                changed = True
                continue
            if new == "tp1" and float(t.get("r_now") or 0) >= 1.0:       # +1R se upar -> 1:1 matlab abhi band
                px = (prices or {}).get(t["symbol"])
                if not px or px.get("tick_age_sec") is None or px["tick_age_sec"] > 60:
                    continue                                               # bhav nahi -> baad mein
                exit_px = px["bid"] if t["bull"] else px["ask"]
                r = ((exit_px - t["entry"]) if t["bull"] else (t["entry"] - exit_px)) / t["risk"]
                self.open = [o for o in self.open if o is not t]
                t["exit"], t["user_exit"] = "tp1", "tp1"
                self._book_close(t, float(r), "manual", pd.Timestamp.now(tz="UTC"))
                self._log_decision(t["symbol"], t, "CLOSED-MANUAL", f"you set 1:1 above +1R -> closed at {exit_px}, R {r:+.2f}")
            else:
                t["exit"], t["user_exit"] = new, new
                t.update(te.levels(t))
                self._log_decision(t["symbol"], t, "EXIT-CHANGED", f"{old} -> {new} by you (R now {t.get('r_now')})")
            control.log_action(self.out, kind="exit_trade", symbol=t["symbol"], direction=t["direction"], zone=_unix(t["zone"]),
                               fill_time=_unix(t["fill_time"]), from_=old, to=new, r_now=t.get("r_now"), score=t.get("score"))
            self.close_done.add(x["id"])
            changed = True
        for x in [x for x in self.control.get("close", []) if x["id"] not in self.close_done]:
            t = next((t for t in self.open if t["symbol"] == x["symbol"] and _unix(t["fill_time"]) == int(x["fill_time"])), None)
            if t is None:
                self.close_done.add(x["id"])                           # pehle hi band ho chuka -> kuch nahi
                self.log(f"  MANUAL: close {x['symbol']} - trade already closed")
                changed = True
                continue
            px = (prices or {}).get(t["symbol"])
            if not px or px.get("tick_age_sec") is None or px["tick_age_sec"] > 60:
                continue                                               # bhav nahi (market band) -> baad mein
            exit_px = px["bid"] if t["bull"] else px["ask"]
            r = ((exit_px - t["entry"]) if t["bull"] else (t["entry"] - exit_px)) / t["risk"]
            self.open = [o for o in self.open if o is not t]
            self._book_close(t, float(r), "manual", pd.Timestamp.now(tz="UTC"))
            self.close_done.add(x["id"])
            self._log_decision(t["symbol"], t, "CLOSED-MANUAL", f"closed by you at {exit_px}, R {r:+.2f}, pnl ${t['pnl']:+.2f}")
            control.log_action(self.out, kind="close", symbol=t["symbol"], direction=t["direction"], zone=_unix(t["zone"]),
                               fill_time=_unix(t["fill_time"]), from_=t["exit"], to="closed", r_now=round(float(r), 3),
                               score=t.get("score"))
            changed = True
        return changed

    def _zone_score(self, k):
        """ML score of a watched zone (for the action log)."""
        for z in self.watch.get(k[0], {}).get("zones", []):
            if z["zone_candle_time"] == k[1] and z["direction"] == k[2]:
                return z.get("score")
        return None

    def _mark_watch_all(self):
        """Watchlist: avoided zones -> status 'avoided'; UNDO hua toh READY wapas (agar bot ne TAKE kiya tha)."""
        for symbol, w in self.watch.items():
            for z in w.get("zones", []):
                av = control.avoid_key(symbol, z["zone_candle_time"], z["direction"]) in self.avoid
                if av and z["status"] in ("ready", "avoided"):
                    z["status"], z["reason"] = "avoided", "avoided manually"
                elif not av and z["status"] == "avoided":
                    z["status"], z["reason"] = "ready", ""
        if self.watch:
            self._write_watchlist()

    def _write_watchlist(self):
        if not self.write_view:
            return
        try:
            _atomic_json(self.out / "watchlist.json", {"updated": pd.Timestamp.now(tz="UTC").isoformat(),
                                                       "threshold": round(float(self.brain.threshold), 3),
                                                       "symbols": self.watch})
        except Exception as e:  # noqa: BLE001
            self.log(f"view: watchlist update mein dikkat ({e.__class__.__name__}: {e})")

    # 3) decisions for the current candle
    def _decide(self, symbol, m5, bar_time):
        self.risk_mode.select(self.cfg, self.equity, self.peak)
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
            wrow = _watch_row(r, o, reason)
            o["zbox"] = {k: wrow[k] for k in ("top", "bottom", "body_top", "body_bottom", "created_at",
                                               "zone_candle_time", "disp_atr", "has_fvg")}   # chart pe zone box
            lab = _category(reason)
            if reason is None and o.get("user") == "take":
                lab = "taken_manual"
            wi.append(dict(o, label=lab, reason=reason or ("taken manually" if lab == "taken_manual" else ""),
                           user="take" if o.get("user") == "take" else ("avoid" if lab == "avoided" else ""),
                           feat=self._features(r),
                           cost_r=float(r.cost_r) * self.cfg.commission_per_lot / wf_commission()
                           if pd.notna(r.cost_r) else None))
            self.watch[symbol]["zones"].append(wrow)
            if reason is None:
                self.armed[symbol].append(o)
                why = "taken manually (you), " if o.get("user") == "take" else ""
                ex = f"{o['exit']}" + (f" (yours, brain {o['exit_bot']})" if o.get("user_exit") else "")
                self._log_decision(symbol, o, "TAKE", f"{why}lots {o['lots']}, risk {o['risk_pct']:.2f}%, {ex}")
            else:
                self._log_decision(symbol, o, "SKIP", reason)
        self.wi_armed[symbol] = wi

    def _features(self, r) -> dict:
        """ML features of this candidate (for the monthly training) - plain python values."""
        out = {}
        for f in self.brain.features:
            v = getattr(r, f, None)
            if isinstance(v, (np.bool_, bool)):
                v = bool(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating, float)):
                v = None if np.isnan(v) else float(v)
            elif v is not None and not isinstance(v, (int, str)):
                v = str(v)
            out[f] = v
        return out

    def _order(self, symbol, r, bar_time, risk_pct, braked):
        bull = r.direction == "bull"
        per_unit = risk_usd_per_unit(symbol, r.entry, r.risk)
        lots = float(np.floor(((self.equity * risk_pct / 100) / per_unit) / LOT / self.cfg.lot_step) * self.cfg.lot_step)
        return dict(symbol=symbol, bar_time=bar_time, zone=r.zone_candle_time, direction=r.direction, bull=bull,
                    entry=float(r.entry), sl=float(r.sl), risk=float(r.risk), sl_pips=round(float(r.sl_pips), 1),
                    score=round(float(r.score), 3), ml_keep=bool(r.ml_keep), va_ok=bool(r.va_ok),
                    va_pos=round(float(r.r24_va_pos), 2) if pd.notna(r.r24_va_pos) else None,
                    act_ok=bool(r.act_ok), act=round(float(r.act_15m), 2) if pd.notna(r.act_15m) else None,
                    exit=self.exit_over.get(control.zone_key(symbol, _unix(r.zone_candle_time), r.direction), r.exit_choice),
                    exit_bot=r.exit_choice,
                    user_exit=self.exit_over.get(control.zone_key(symbol, _unix(r.zone_candle_time), r.direction), ""),
                    lots=round(lots, 2), risk_usd=round(lots * LOT * per_unit, 2),
                    usd_side=usd_side(symbol, r.direction), braked=braked, risk_pct=round(risk_pct, 3),
                    spread_live=_pips(symbol, getattr(r, "spread_live", None)),
                    spread_used=_pips(symbol, getattr(r, "spread_used", None)))

    def _skip_reason(self, symbol, r, o, today, bar_time):
        taken = control.zone_key(symbol, _unix(o["zone"]), o["direction"]) in self.take
        if taken and not (o["ml_keep"] and o["va_ok"] and o["act_ok"]):
            o["user"] = "take"                     # tera TAKE: ML / value area / quiet ko override (risk rules nahi)
        elif not o["ml_keep"]:
            return f"ML score {o['score']:.2f} < {self.brain.threshold:.2f}"
        elif not o["va_ok"]:
            return f"price above value area ({o['va_pos']})"
        elif not o["act_ok"]:
            return f"market quiet (activity {o['act']} < {config.ACT_MIN})"
        if self.kill:
            return "kill switch active"
        if self.news is not None and self.news.blocked(symbol, bar_time):
            return f"news window: {self.news.blocked(symbol, bar_time)}"
        why = daily_ok(self.cfg, self.day_start[today], self.day_pnl.get(today, 0.0), self._open_risk(),
                       o["risk_usd"] + self.cfg.commission_per_lot * o["lots"])
        if why:
            return why
        zk = (symbol, str(o["zone"]), o["direction"])          # same zone already traded (e.g. filled + closed this candle)
        if any((t["symbol"], str(t["zone"]), t["direction"]) == zk for t in self.open + self.closed[-50:]):
            return "zone already traded"
        busy = self.open                        # backtest jaisa: sirf KHULE trades gine jaate hain
        if len(busy) >= self.cfg.max_open:
            return "max open trades"
        if any(b["symbol"] == symbol for b in busy):
            return "already in this pair"
        if sum(b["usd_side"] == o["usd_side"] for b in busy) >= self.cfg.max_usd_same_side:
            return "too much USD one way"
        if o["lots"] < self.cfg.min_lot:
            return "lot below 0.01"
        if self._avoided(symbol, o):              # bot lena chahta tha, par tune dashboard se mana kiya
            return "avoided manually"
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
                label = "taken_manual" if label == "taken_manual" else "taken"
            elif label in ("taken", "taken_manual"):
                # TAKE tha, par fill ke waqt limit / daily rule ne roka - ya candle ke beech tune AVOID dabaya
                label = "avoided" if self._avoided(symbol, o) else "fill_skipped"
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
                       net_brain=round(net[w["exit"]], 3), user=w.get("user", ""),
                       exit_bot=w.get("exit_bot", w["exit"]), net_bot=round(net[w.get("exit_bot", w["exit"])], 3))
            pd.DataFrame([row]).reindex(columns=WI_COLS).to_csv(self.wi_file, mode="a", header=not self.wi_file.exists(),
                                                                index=False)
            try:                                   # mahine ki training ke liye: features + result (src/ml/monthly_retrain.py)
                rec = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in row.items()}
                rec["features"] = w.get("feat") or {}
                with self.feat_file.open("a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            except Exception as e:  # noqa: BLE001
                self.log(f"what-if features log: {e}")
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
        self.risk_mode = RiskMode(aggressive=st.get('risk_mode', 'aggressive') == 'aggressive')
        self.month, self.month_start = st.get("month"), float(st.get("month_start", self.equity))
        from datetime import date
        self.day_start = {date.fromisoformat(k): float(v) for k, v in st.get("day_start", {}).items()}
        self.day_pnl = {date.fromisoformat(k): float(v) for k, v in st.get("day_pnl", {}).items()}
        self.open = [_from_json(t) for t in st.get("open_trades", [])]
        self.wi_open = [_from_json(w) for w in st.get("whatif_open", [])]
        self.n_closed_before = int(st.get("closed", 0))
        self.close_done = set(st.get("close_done", []))
        self.log(f"state: purana hisaab load (saved {st.get('updated')}): equity ${self.equity:,.2f}, peak ${self.peak:,.2f}, "
                 f"{len(self.open)} khule trade, {len(self.wi_open)} what-if, kill switch {'ON' if self.kill else 'off'}. "
                 f"Band rehte hue jo hua woh pehli candle pe update hoga.")

    def save(self):
        self.risk_mode.select(self.cfg, self.equity, self.peak)
        hist = pd.read_csv(self.closed_file) if self.closed_file.exists() else pd.DataFrame()
        rows = [dict(_to_json(t), r=t.get("r_now")) for t in self.open]
        snap = pd.concat([hist.assign(status="closed"),
                          pd.DataFrame(rows).reindex(columns=CLOSED_COLS).assign(status="open")], ignore_index=True) \
            if rows else hist.assign(status="closed")
        if len(snap):
            snap.to_csv(self.out / "shadow_trades.csv", index=False)
        pnl = hist["pnl"] if "pnl" in hist else pd.Series(dtype=float)
        state = dict(version=STATE_VERSION, equity=round(self.equity, 2), peak=round(self.peak, 2), kill=self.kill,
                     risk_mode='aggressive' if self.risk_mode.aggressive else 'conservative',
                     month=self.month, month_start=round(self.month_start, 2),
                     day_start={str(k): round(v, 2) for k, v in self.day_start.items()},
                     day_pnl={str(k): round(v, 2) for k, v in self.day_pnl.items()},
                     open_trades=[_to_json(t) for t in self.open],
                     whatif_open=[_to_json(w) for w in self.wi_open],
                     close_done=sorted(self.close_done)[-200:],
                     open=len(self.open), closed=int(len(hist)),
                     win_pct=round(100 * float((pnl > 0).mean()), 1) if len(pnl) else None,
                     risk=asdict(self.cfg), updated=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str))
        tmp.replace(self.state_file)               # atomic: crash ke beech file kharab nahi hoti


def reset_state(out: Path = LIVE):
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    for f in ("shadow_state.json", "closed_trades.csv", "shadow_trades.csv", "whatif.csv", "traded_zones.json",
              "control.json"):
        p = out / f
        if p.exists():
            p.replace(out / f"{p.stem}_old_{stamp}{p.suffix}")
    print(f"reset: purane files '_old_{stamp}' naam se rakh diye. Ab $10,000 se fresh shuru.")


def run_live(poll_s: float = 1.0, health_every_s: float = 5.0):
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
    orb = None
    try:                                         # ORB shadow (src/live/orb.py): alag virtual account, zone bot pe asar nahi
        from src.live.orb import OrbEngine
        orb = OrbEngine(feed, resume=True, news_guard=guard)
    except Exception as e:  # noqa: BLE001 - ORB na chale toh bhi zone bot chalta rahe
        print(f"ORB off: {e.__class__.__name__}: {e}")
    last = {}
    news_hour = pd.Timestamp.now(tz="UTC").floor("h")
    last_health = 0.0
    last_week = 0.0
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
            prices = monitor.write_ticks(feed, eng, LIVE)              # dashboard: live bid/ask + candle (har ~1 s)
            if eng.apply_control(prices):                              # dashboard: AVOID / UNDO / CLOSE buttons
                eng.save()
            if prices and eng.fill_intrabar(prices):                   # limit chhua -> trade turant khula
                eng.save()                                             # dashboard ko seconds mein dikhe
            if eng.open and eng.manage_ticks():                        # SL / TP / trail har tick pe -> close turant
                eng.save()
            if eng._redecide and eng.redecide_pending():               # trade band -> watchlist / orders turant update
                eng.save()
            if orb is not None:
                try:
                    if orb.step():                                     # ORB: dabba / break / trade har ~1 s
                        orb.save()
                except Exception as e:  # noqa: BLE001
                    print(f"ORB step error: {e.__class__.__name__}: {e}")
            if time.time() - last_health >= health_every_s:
                monitor.write_health(feed, eng, LIVE, started_at, guard)   # dashboard: MT5 / VPS sehat (har ~5 s)
                last_health = time.time()
            if time.time() - last_week >= 60:                          # Saturday (NY): weekend report (once)
                last_week = time.time()
                try:
                    from src.live import weekly_report
                    if weekly_report.maybe_write(LIVE):
                        print("weekend report written -> data/live/weekly/")
                except Exception as e:  # noqa: BLE001 - report kabhi bot na roke
                    print(f"weekly report error: {e.__class__.__name__}: {e}")
            time.sleep(poll_s)
    except KeyboardInterrupt:
        eng.save()
        if orb is not None:
            orb.save()
        feed.close()
        print("Shadow mode band. Files: data/live/decisions.csv, shadow_trades.csv, shadow_state.json")


def run_replay(start: str, end: str, out: Path):
    from src.live.feeds import ReplayFeed
    feed = ReplayFeed(config.CORE_SYMBOLS, start, end, HISTORY_DAYS)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("decisions.csv", "shadow_trades.csv", "closed_trades.csv", "whatif.csv", "shadow_state.json",
              "orb_state.json", "orb_trades.csv", "orb_decisions.csv"):
        (out / f).unlink(missing_ok=True)
    eng = ShadowEngine(feed, out_dir=out, spread_floor=False, write_view=False)   # Dukascopy = training data, floor nahi
    from src.live.orb import OrbEngine
    orb = OrbEngine(feed, out_dir=out, resume=False, write_view=False)
    t0 = time.time()
    for i, _ in enumerate(feed):
        for s in eng.symbols:
            eng.step(s)
        orb.step()
    eng.save()
    orb.save()
    print(f"ORB replay: closed {orb.n_closed}, open {len(orb.open)}, ORB equity ${orb.equity:,.2f}")
    print(f"replay {start} -> {end}: {len(feed.times)} candles x {len(eng.symbols)} pairs in {time.time() - t0:.0f}s | "
          f"trades closed {len(eng.closed)}, open {len(eng.open)}, equity ${eng.equity:,.2f}")
    return eng


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shadow mode (no orders)")
    ap.add_argument("--replay", nargs=2, metavar=("START", "END"), help="purane data pe chalao, e.g. 2026-09-21 2026-09-26")
    ap.add_argument("--reset", action="store_true", help="purana hisaab hata ke $10,000 se fresh shuru")
    ap.add_argument("--reset-orb", action="store_true", help="sirf ORB ka hisaab fresh (zone bot ka nahi)")
    a = ap.parse_args()
    if a.reset and not a.replay:
        reset_state()
    if (a.reset or a.reset_orb) and not a.replay:
        from src.live.orb import reset_state as reset_orb
        reset_orb()
    if a.replay:
        run_replay(a.replay[0], a.replay[1], LIVE / "replay")
    else:
        run_live()
