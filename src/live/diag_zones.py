r"""
DIAGNOSIS (sirf padhta hai, koi order / koi file change nahi): ek pair ke haal ke M15 zones aur
har zone watchlist mein kyun nahi hai (touch ho gaya / mar gaya / purana).

Chalana (VPS pe, venv active, MT5 khula; bot chalta rahe toh bhi theek hai):
    python src\live\diag_zones.py              (GBPUSD, last 3 din)
    python src\live\diag_zones.py EURUSD --days 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.features.smc import find_orderflow_zones  # noqa: E402
from src.live.live_features import HISTORY_DAYS, SL_BUFFER_ATR, frames  # noqa: E402

NY = "America/New_York"


def ny(t):
    return pd.Timestamp(t).tz_convert(NY).strftime("%a %d %b %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="GBPUSD")
    ap.add_argument("--days", type=float, default=3)
    a = ap.parse_args()
    sym = a.symbol.upper()

    from src.live.feeds import MT5Feed
    feed = MT5Feed(HISTORY_DAYS)
    try:
        m5, bar_time = feed.bars(sym)
    finally:
        feed.close()
    print(f"\n{sym}: {len(m5)} band M5 candles, aakhri {ny(m5.index[-1])} NY, abhi wali candle {ny(bar_time)} NY")
    print(f"aakhri 3 candles:\n{m5.tail(3)[['open', 'high', 'low', 'close', 'volume', 'spread']]}")

    m15 = frames(m5, bar_time)["M15"]
    disp = config.disp_for(sym)
    since = bar_time - pd.Timedelta(days=a.days)
    for d in sorted({disp, 1.5, 2.0}):
        z = find_orderflow_zones(m15, swing_n=config.SWING_N, min_disp_atr=d)
        n = int((z["created_at"] >= since).sum()) if len(z) else 0
        print(f"displacement {d}x{' (BOT SETTING)' if d == disp else ''}: {n} zones in last {a.days:g} days")

    z = find_orderflow_zones(m15, swing_n=config.SWING_N, min_disp_atr=disp)
    z = z[z["created_at"] >= since].sort_values("created_at")
    lo, hi, cl, sp = (m5[c].to_numpy() for c in ["low", "high", "close", "spread"])
    idx, f = m5.index, len(m5)
    print(f"\nZones (bot setting {disp}x) - watchlist mein sirf 'ALIVE, untouched' wale aate hain:")
    if z.empty:
        print("  koi zone nahi bana -> market mein itna strong displacement (structure break) nahi aaya")
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        entry = config.entry_price(r, bull)
        sl = (r.bottom - SL_BUFFER_ATR * r.atr) if bull else (r.top + SL_BUFFER_ATR * r.atr)
        s = int(idx.searchsorted(r.created_at))
        dead = (cl[s:f] < r.bottom) if bull else (cl[s:f] > r.top)
        touched = (lo[s:f] + sp[s:f] <= entry) if bull else (hi[s:f] >= entry)
        if dead.any():
            st = f"DEAD at {ny(idx[s + dead.argmax()])} (candle closed beyond the zone)"
        elif touched.any():
            st = f"TOUCHED at {ny(idx[s + touched.argmax()])} (price reached entry -> bot decided then)"
        else:
            st = "ALIVE, untouched -> should be in watchlist"
        print(f"  {'BUY ' if bull else 'SELL'} zone candle {ny(r.zone_candle_time)} | created {ny(r.created_at)} | "
              f"entry {entry:.5f} SL {sl:.5f} ({abs(entry - sl) / 0.0001:.1f} pips) | {st}")

    st = ROOT / "data" / "live" / "shadow_state.json"
    if st.exists():
        s = json.loads(st.read_text())
        mine = [t for t in s.get("open_trades", []) if t.get("symbol") == sym]
        print(f"\nshadow_state: {len(mine)} khula {sym} trade(s)" + "".join(
            f"\n  {t['direction']} {t['entry']} fill {t['fill_time']} exit {t['exit']}" for t in mine))
    dec = ROOT / "data" / "live" / "decisions.csv"
    if dec.exists():
        d = pd.read_csv(dec)
        d = d[d["symbol"] == sym].tail(8)
        print(f"\ndecisions.csv - {sym} ke aakhri {len(d)} decisions:")
        for r in d.itertuples(index=False):
            print(f"  {r.bar_time} {r.direction} {r.entry} ML {r.score} -> {r.decision}: {r.reason}")


if __name__ == "__main__":
    main()
