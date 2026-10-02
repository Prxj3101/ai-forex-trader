r"""
MARKET ACTIVITY at the zone touch (ACTIVITY FILTER, 2 Oct 2026).

act_15m = average tick volume of the last 3 M5 candles BEFORE the fill candle
          / average tick volume of the last 288 M5 candles (24h) before the fill candle
  > 1  = market busier than usual right now
  < 1  = market quieter than usual (e.g. dead Asia hour, lunch, rollover)

Why: real CME orderflow test (src/research/orderflow_test.py, orderflow_test2.py) showed that delta /
absorption / big orders add nothing robust, but "is the market active when price taps the zone" does:
quiet touches lose (PF ~0.9), active touches win. Free tick volume measures it as well as CME volume.

Only candles before the fill candle are used (no lookahead) -> backtest and live give the same number.
Volume is a RATIO (now vs last 24h), so a different broker's tick volume level does not matter much.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RECENT_BARS = 3        # 15 min
BASE_BARS = 288        # 24h


def act_at(volume: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """volume: M5 tick volume array; idx: position of the FILL bar (only bars < idx are used)."""
    cv = np.concatenate([[0.0], np.cumsum(np.asarray(volume, dtype=float))])
    i = np.asarray(idx, dtype=np.int64)
    a = np.clip(i - RECENT_BARS, 0, None)
    b = np.clip(i - BASE_BARS, 0, None)
    recent = (cv[i] - cv[a]) / np.maximum(i - a, 1)
    base = (cv[i] - cv[b]) / np.maximum(i - b, 1)
    ok = (i - b >= BASE_BARS // 2) & (base > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, recent / base, np.nan)


def add_activity(d: pd.DataFrame, symbol: str, m5: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds act_15m to trades of ONE symbol (needs fill_time)."""
    if m5 is None or "volume" not in m5:
        m5 = pd.read_parquet(DATA / f"{symbol}_M5.parquet", columns=["volume"])
    d = d.copy()
    if d.empty:
        d["act_15m"] = pd.Series(dtype=float)
        return d
    idx = m5.index.searchsorted(d["fill_time"])
    d["act_15m"] = act_at(m5["volume"].to_numpy(float), idx)
    return d
