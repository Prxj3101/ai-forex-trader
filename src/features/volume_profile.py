"""
Volume Profile features (where did trading happen, by PRICE).

Volume Profile = for a time range, how much volume traded at each price level.
  POC (point of control) = price with the most volume ("fair price", acts like a magnet)
  Value Area (VA)        = price band holding 70% of the volume (VAL = low edge, VAH = high edge)
  HVN / LVN              = high / low volume areas (price stalls in HVN, moves fast through LVN)

We build it ourselves from Dukascopy M5 candles: each candle's volume is spread evenly over
its high-low range, in 1-pip price bins. (Tick volume, not real exchange volume - like OANDA on
TradingView.)

Two profiles per trade, both from bars BEFORE the fill bar (no lookahead):
  pd_  = previous NY trading day (5 PM -> 5 PM), the classic "yesterday's profile"
  r24_ = last 24 hours before the fill (includes the current session)

Features (direction-adjusted: positive = towards the trade's profit side):
  {p}_poc_dist_atr : distance entry -> POC in ATR. >0 = POC is in the profit direction
                     (bull: POC above entry = entry is "cheap")
  {p}_poc_dist_r   : same distance in R (risk units) -> could the POC be a target?
  {p}_va_pos       : where entry sits in the value area, 0 = edge on the "cheap" side for this
                     trade (VAL for a buy), 1 = other edge; <0 / >1 = outside the VA
  {p}_zone_density : volume inside the zone vs average of the profile (<1 = thin area / LVN)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}
VP_FEATURES = [f"{p}_{f}" for p in ("pd", "r24") for f in ("poc_dist_atr", "poc_dist_r", "va_pos", "zone_density")]


def _profile(lo_b: np.ndarray, hi_b: np.ndarray, vol: np.ndarray):
    """Histogram over bins [min..max] with each bar's volume spread evenly over its bins.
    Returns (first_bin, hist) or None."""
    if len(vol) == 0 or vol.sum() <= 0:
        return None
    b0 = int(lo_b.min())
    n = int(hi_b.max()) - b0 + 1
    per = vol / (hi_b - lo_b + 1)
    diff = np.zeros(n + 1)
    np.add.at(diff, lo_b - b0, per)
    np.add.at(diff, hi_b - b0 + 1, -per)
    return b0, np.cumsum(diff)[:n]


def _stats(b0: int, h: np.ndarray):
    poc = b0 + int(np.argmax(h))
    order = np.argsort(h)[::-1]
    k = int(np.searchsorted(np.cumsum(h[order]), 0.7 * h.sum())) + 1
    va = order[:k]
    return poc, b0 + int(va.min()), b0 + int(va.max())


def _features(prof, entry_b, top_b, bot_b, bull, atr_b, risk_b):
    if prof is None:
        return [np.nan] * 4
    b0, h = prof
    poc, val, vah = _stats(b0, h)
    sgn = 1 if bull else -1
    dist = (poc - entry_b) * sgn
    width = max(vah - val, 1)
    pos = (entry_b - val) / width if bull else (vah - entry_b) / width
    a, b = max(bot_b - b0, 0), min(top_b - b0, len(h) - 1)
    mean_all = h[h > 0].mean()
    dens = (h[a:b + 1].mean() / mean_all) if (b >= a and mean_all > 0) else 0.0
    return [dist / atr_b, dist / risk_b, pos, dens]


def add_vp_features(d: pd.DataFrame, symbol: str, m5: pd.DataFrame | None = None) -> pd.DataFrame:
    if m5 is None:
        m5 = pd.read_parquet(DATA / f"{symbol}_M5.parquet", columns=["high", "low", "volume"])
    pip = PIP[symbol]
    lo_b = np.floor(m5["low"].to_numpy() / pip).astype(np.int64)
    hi_b = np.floor(m5["high"].to_numpy() / pip).astype(np.int64)
    vol = m5["volume"].to_numpy(float)
    idx = m5.index

    # NY trading day of every M5 bar (day starts 17:00 NY)
    ny = idx.tz_convert("America/New_York")
    tday = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    day_start = pd.Series(np.arange(len(idx))).groupby(tday.values).min()
    day_end = pd.Series(np.arange(len(idx))).groupby(tday.values).max() + 1
    days = day_start.index.to_numpy()

    out = np.full((len(d), 8), np.nan)
    f0s = idx.searchsorted(pd.DatetimeIndex(d["fill_time"]))
    cache: dict[int, tuple] = {}
    for i, (r, f0) in enumerate(zip(d.itertuples(index=False), f0s)):
        bull = r.direction == "bull"
        entry_b = int(np.floor(r.entry / pip))
        top_b, bot_b = int(np.floor(r.top / pip)), int(np.floor(r.bottom / pip))
        atr_b, risk_b = max(r.atr / pip, 1e-9), max(r.risk / pip, 1e-9)

        # previous NY trading day
        fd = (pd.Timestamp(r.fill_time).tz_convert("America/New_York") + pd.Timedelta(hours=7)).normalize().tz_localize(None)
        k = int(np.searchsorted(days, np.datetime64(fd))) - 1
        if k >= 0:
            if k not in cache:
                a, b = int(day_start.iloc[k]), int(day_end.iloc[k])
                cache[k] = _profile(lo_b[a:b], hi_b[a:b], vol[a:b])
            out[i, :4] = _features(cache[k], entry_b, top_b, bot_b, bull, atr_b, risk_b)
        # rolling last 24h (288 M5 bars) before the fill bar
        a = max(f0 - 288, 0)
        out[i, 4:] = _features(_profile(lo_b[a:f0], hi_b[a:f0], vol[a:f0]), entry_b, top_b, bot_b, bull, atr_b, risk_b)

    d = d.copy()
    for j, c in enumerate(VP_FEATURES):
        d[c] = out[:, j]
    return d
