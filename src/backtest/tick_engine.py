r"""
TICK ENGINE: one trade, replayed on real ticks (Dukascopy, data/ticks/store).

Used by the corrected backtest / ML labels (and the same rules are used by the live shadow bot).

Rules (fixed BEFORE looking at results):
  FILL   BUY limit fills at the first tick with ask <= entry, SELL limit with bid >= entry,
         searched from `search_from` to `search_to`. Fill price = entry (no positive slippage).
         Price movement BEFORE the fill tick is ignored completely.
  EXITS  checked on every tick after the fill. BUY exits on bid, SELL exits on ask.
         SL / trailing stop: first tick at or beyond the stop -> exit at THAT tick's price
         (gaps / spikes give a worse price, never better than the stop).
         TP (limit): first tick at or beyond TP -> exit exactly at TP.
         Same tick can never be both (SL < entry < TP); tick order decides everything.
  TRAIL  'trail' = after +1R the stop follows the best price 1R behind (best - 1R).
         The bot updates the stop once per second: a new best only moves the stop for ticks
         at least 1 second later; until then the old stop stays active.
  HOLD   max 3 days after the fill (same as before) -> close at the last tick price.
  R      gross R = (exit price - entry) / (entry - SL) (sign by direction). Costs are added later.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "ticks" / "store"
HOLD_MS = 3 * 24 * 3600 * 1000
TRAIL_DELAY_MS = 1000


@lru_cache(maxsize=6)
def _month(sym: str, ym: str):
    f = STORE / sym / f"{ym}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["t"], z["ask"], z["bid"]


def ticks(sym: str, t0_ms: int, t1_ms: int):
    """All ticks with t0 <= t < t1 (UTC ms). Returns (t, ask, bid) or None if a month file is missing."""
    months = pd.period_range(pd.Timestamp(t0_ms, unit="ms"), pd.Timestamp(max(t1_ms - 1, t0_ms), unit="ms"), freq="M")
    ts, asks, bids = [], [], []
    for m in months:
        z = _month(sym, str(m))
        if z is None:
            return None
        t, a, b = z
        i, j = np.searchsorted(t, t0_ms), np.searchsorted(t, t1_ms)
        ts.append(t[i:j]); asks.append(a[i:j]); bids.append(b[i:j])
    return np.concatenate(ts), np.concatenate(asks), np.concatenate(bids)


def find_fill(sym, bull, entry, search_from_ms, search_to_ms):
    """UTC ms of the fill tick, or None (never filled) / 'missing' (no tick data)."""
    z = ticks(sym, search_from_ms, search_to_ms)
    if z is None:
        return "missing"
    t, a, b = z
    hit = np.nonzero(a <= entry)[0] if bull else np.nonzero(b >= entry)[0]
    return int(t[hit[0]]) if len(hit) else None


def run_exit(sym, bull, entry, sl, fill_ms, policy: str, tp_r: float | None = None):
    """policy 'tp1' / 'tp2' (fixed 1:1 / 1:2), 'tp' (fixed tp_r) or 'trail' (trail 1R after 1R).
    Returns dict(r, reason, exit_ms, best_r)."""
    z = ticks(sym, fill_ms, fill_ms + HOLD_MS)
    if z is None:
        return dict(r=np.nan, reason="missing", exit_ms=None, best_r=np.nan)
    return exit_arrays(*z, bull, entry, sl, policy, tp_r)


def exit_arrays(t, a, b, bull, entry, sl, policy: str, tp_r: float | None = None):
    """Same as run_exit but on tick arrays that START at the fill tick (already cut to the 3-day hold)."""
    risk = abs(entry - sl)
    px = b if bull else a                                   # exit side
    fav = (px - entry) / risk if bull else (entry - px) / risk
    n = len(fav)
    if n == 0:
        return dict(r=np.nan, reason="missing", exit_ms=None, best_r=np.nan)
    if policy in ("tp1", "tp2", "tp"):
        tp = {"tp1": 1.0, "tp2": 2.0}.get(policy, tp_r)
        sl_hit, tp_hit = fav <= -1.0, fav >= tp
        ia = int(np.argmax(sl_hit)) if sl_hit.any() else n
        ib = int(np.argmax(tp_hit)) if tp_hit.any() else n
        if ia < n and ia < ib:
            return dict(r=float(fav[ia]), reason="sl", exit_ms=int(t[ia]), best_r=float(fav[:ia + 1].max()))
        if ib < n:
            return dict(r=tp, reason="tp", exit_ms=int(t[ib]), best_r=tp)
    elif policy == "trail":
        best = np.maximum.accumulate(fav)
        raw = np.where(best >= 1.0, best - 1.0, -1.0)         # stop the bot WANTS after tick i
        lag = np.searchsorted(t, t - TRAIL_DELAY_MS, side="right") - 1   # last tick at least 1 s earlier
        stop = np.where(lag >= 0, raw[np.clip(lag, 0, None)], -1.0)     # stop ACTIVE at tick i
        hit = fav <= stop
        if hit.any():
            i = int(np.argmax(hit))
            s = stop[i]
            why = "sl" if s < -0.01 else ("be" if abs(s) <= 0.01 else "trail")
            return dict(r=float(fav[i]), reason=why, exit_ms=int(t[i]), best_r=float(best[i]))
    else:
        raise ValueError(policy)
    return dict(r=float(fav[-1]), reason="timeout", exit_ms=int(t[-1]), best_r=float(fav.max()))


# ----------------------------------------------------------------------
# STATEFUL version (live shadow bot + replay): same rules as run_exit, but fed tick batches as they arrive.
# A trade dict carries: bull, entry, risk, exit ('tp1' | 'trail'), entry_ms, and the state below.
def init_state(t: dict, entry_ms: int) -> dict:
    t.update(entry_ms=int(entry_ms), last_ms=int(entry_ms) - 1, best_r=0.0, stop_r=-1.0,
             pending_stop=None, pending_from=None, r_now=0.0)
    return t


def feed_ticks(t: dict, tt, ask, bid):
    """Process ticks (UTC ms, ask, bid) newer than t['last_ms']. Returns (r, reason, exit_ms) when the trade
    closes, else None (state updated in place). Rules = run_exit: exit side bid (BUY) / ask (SELL), stop at the
    real tick price, TP exactly, trail stop moves 1 s after a new best, max hold 3 days."""
    bull, entry, risk = t["bull"], float(t["entry"]), float(t["risk"])
    trail = t["exit"] == "trail"
    tp = None if trail else (2.0 if t["exit"] == "tp2" else 1.0)
    hold_end = t["entry_ms"] + HOLD_MS
    for ms, a, b in zip(tt, ask, bid):
        ms = int(ms)
        if ms <= t["last_ms"]:
            continue
        if ms >= hold_end:
            return t["r_now"], "timeout", t["last_ms"]
        t["last_ms"] = ms
        px = b if bull else a
        fav = ((px - entry) if bull else (entry - px)) / risk
        t["r_now"] = round(float(fav), 3)
        if t["pending_stop"] is not None and ms >= t["pending_from"]:
            t["stop_r"] = max(t["stop_r"], t["pending_stop"])
            t["pending_stop"] = t["pending_from"] = None
        if fav <= t["stop_r"]:
            s = t["stop_r"]
            return float(fav), ("sl" if s < -0.01 else ("be" if abs(s) <= 0.01 else "trail")), ms
        if tp is not None and fav >= tp:
            return float(tp), "tp", ms
        if fav > t["best_r"]:
            t["best_r"] = float(fav)
            if trail and fav >= 1.0:
                t["pending_stop"], t["pending_from"] = float(fav) - 1.0, ms + TRAIL_DELAY_MS
    return None


def levels(t: dict) -> dict:
    """Chart levels from the live state: TP (fixed exits) and the stop that is active now."""
    sign = 1 if t["bull"] else -1
    entry, risk = float(t["entry"]), float(t["risk"])
    tp = None if t["exit"] == "trail" else (2.0 if t["exit"] == "tp2" else 1.0)
    return dict(best_r=round(float(t.get("best_r", 0.0)), 3),
                stop_now=round(entry + sign * float(t.get("stop_r", -1.0)) * risk, 6),
                tp=round(entry + sign * tp * risk, 6) if tp is not None else None)
