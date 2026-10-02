r"""
SPREAD FLOOR: ML ko hamesha "training jaisa" spread dikhana.

Problem (3 Oct 2026 shadow check): ML ne Dukascopy data pe seekha (EURUSD ~0.3 pip, USDJPY ~0.4).
Live broker kabhi 0.0-0.1 pip dikhata hai (MetaQuotes demo, Funding Pips JPY). ML ka sabse
important feature spread_r (spread / SL) hai -> itna chhota spread ML ne kabhi dekha hi nahi ->
score galat upar chala jaata hai -> kamzor trades pass ho jaate hain (2 Oct: 3 extra EURUSD losses).

Fix: ML ke features ke liye spread = max(live spread, Dukascopy ka normal spread us pair + NY ghante ka).
  - live spread bada ho (news, rollover) -> asli bada spread hi dikhta hai (ML ne training mein dekha hai)
  - live spread nakli/bahut chhota ho -> normal (median) spread dikhta hai
Fill check aur P&L ASLI live spread se hi hote hain (yeh file sirf ML ke input ke liye hai).

Table: src/live/spread_floor.json  {symbol: [24 values, NY hour 0..23, price units]}
Banane ke liye (Dukascopy data update hone ke baad kabhi kabhi):
    python src\live\spread_floor.py
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

DATA = ROOT / "data" / "processed"
FILE = Path(__file__).with_name("spread_floor.json")
NY = "America/New_York"
DAYS = 180                  # pichhle ~6 mahine ka median (abhi ka "normal" spread)
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}

_cache: dict | None = None


def build(symbols=None, days: int = DAYS) -> dict:
    """Dukascopy M5 se har pair ka har NY ghante ka median spread."""
    out = {}
    for s in symbols or config.CORE_SYMBOLS:
        sp = pd.read_parquet(DATA / f"{s}_M5.parquet", columns=["spread"])["spread"]
        sp = sp[sp.index >= sp.index.max() - pd.Timedelta(days=days)]
        hour = sp.index.tz_convert(NY).hour
        med = sp.groupby(hour).median().reindex(range(24))
        med = med.fillna(sp.median())
        out[s] = [round(float(v), 6) for v in med]
    return out


def save(table: dict) -> None:
    FILE.write_text(json.dumps(table, indent=1))


def load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(FILE.read_text()) if FILE.exists() else {}
    return _cache


def apply(m5: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Copy of m5 jisme spread = max(spread, floor[NY hour]). Table na ho toh m5 waisa hi."""
    floor = load().get(symbol)
    if not floor:
        return m5
    hours = m5.index.tz_convert(NY).hour
    f = np.asarray(floor)[hours]
    out = m5.copy()
    out["spread"] = np.maximum(out["spread"].to_numpy(float), f)
    return out


if __name__ == "__main__":
    t = build()
    save(t)
    print(f"saved {FILE}")
    for s, v in t.items():
        p = PIP[s]
        print(f"  {s}: " + " ".join(f"{h:02d}h {x / p:.1f}" for h, x in enumerate(v)))
