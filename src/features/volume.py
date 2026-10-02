"""
Tick-volume features (free "orderflow proxy" from Dukascopy M5 volume).

Forex has no central exchange, so this is NOT real orderflow (no buy/sell side, no delta).
Dukascopy volume = traded volume on THEIR platform only. Its level also jumps between years
(e.g. 2021 is ~3x lower than 2020), so raw volume is useless -> we always use RELATIVE volume:

    rvol = volume of the bars / normal volume for the SAME time of day (NY clock)
           over the previous 20 trading days (median, only past data = no lookahead)

    rvol 2.0 = twice the usual activity for that time of day.

Features (per trade, all from bars BEFORE the fill bar):
    zone_rvol      : activity in the zone candle itself
    disp_rvol      : activity during the displacement (after zone candle until BOS candle closed)
    appr_rvol_1h   : activity in the last 1h before the fill (price coming into the zone)
    appr_rvol_3h   : same, last 3h
    pre_fill_rvol  : last 15 min before the fill
    vol_regime     : market activity last 5 days vs last 60 days (busy or quiet market)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
TF_MIN = {"M15": 15, "H1": 60, "H4": 240}
VOL_FEATURES = ["zone_rvol", "disp_rvol", "appr_rvol_1h", "appr_rvol_3h", "pre_fill_rvol", "vol_regime"]


def baseline(m5: pd.DataFrame, days: int = 20) -> pd.DataFrame:
    """Adds 'base' = median volume of the same NY time-of-day slot over the previous `days` days."""
    m = m5[["volume"]].copy()
    ny = m.index.tz_convert("America/New_York")
    slot = ny.hour * 60 + ny.minute
    m["base"] = (m.groupby(slot)["volume"]
                  .transform(lambda s: s.shift(1).rolling(days, min_periods=5).median()))
    daily = m["volume"].groupby(ny.date).sum()
    reg = (daily.shift(1).rolling(5).mean() / daily.shift(1).rolling(60, min_periods=20).mean())
    m["regime"] = pd.Series(ny.date, index=m.index).map(reg).to_numpy()
    return m


def _ratio(cv: np.ndarray, cb: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sum(volume[a:b]) / sum(base[a:b]) using cumulative sums (vectorised)."""
    a = np.clip(a, 0, len(cv) - 1)
    b = np.clip(b, 0, len(cv) - 1)
    v = cv[b] - cv[a]
    base = cb[b] - cb[a]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((b > a) & (base > 0), v / base, np.nan)
    return r


def add_volume_features(d: pd.DataFrame, symbol: str, m5: pd.DataFrame | None = None) -> pd.DataFrame:
    """d = zone_dataset_v2 rows of ONE symbol. Returns d with VOL_FEATURES columns added."""
    if m5 is None:
        m5 = pd.read_parquet(DATA / f"{symbol}_M5.parquet", columns=["volume"])
    m = baseline(m5)
    vol = m["volume"].to_numpy(float)
    base = np.nan_to_num(m["base"].to_numpy(float), nan=0.0)
    vol = np.where(base > 0, vol, 0.0)                     # bars without a baseline count as nothing
    cv = np.concatenate([[0.0], np.cumsum(vol)])           # cv[i] = sum of bars [0, i)
    cb = np.concatenate([[0.0], np.cumsum(base)])
    idx = m.index

    def pos(t):  # first M5 bar at or after t
        return idx.searchsorted(pd.DatetimeIndex(t))

    d = d.copy()
    tf_min = d["zone_tf"].astype(str).map(TF_MIN).fillna(15).to_numpy()
    z0 = pos(d["zone_candle_time"])
    z1 = pos(d["zone_candle_time"] + pd.to_timedelta(tf_min, unit="m"))
    bos_end = pos(d["bos_time"] + pd.to_timedelta(tf_min, unit="m"))
    f0 = pos(d["fill_time"])

    d["zone_rvol"] = _ratio(cv, cb, z0, z1)
    d["disp_rvol"] = _ratio(cv, cb, z1, bos_end)
    d["appr_rvol_1h"] = _ratio(cv, cb, f0 - 12, f0)
    d["appr_rvol_3h"] = _ratio(cv, cb, f0 - 36, f0)
    d["pre_fill_rvol"] = _ratio(cv, cb, f0 - 3, f0)
    d["vol_regime"] = m["regime"].to_numpy()[np.clip(f0, 0, len(m) - 1)]
    return d
