r"""
GOLD FIB 7 - ENTRY x SL GRID (15 Oct 2026). He asked: entry at 3 places of the 0.705-0.786 zone (top edge, middle,
lower edge) x SL options, see which combination works. He himself puts SL just below/past 0.786.
SETS     M15 K2.5 | H1 K2.5 | H1 K3.5 zones (zones_v2: ATR zigzag, fresh, H4 trend), 2016-2026.
ENTRY    E705 = 0.705 (top edge) | E745 = 0.7455 (middle) | E786 = 0.786 (lower edge). Limit, FIRST chart touch of
         that level (bid crosses it), fill at that tick's real price (BUY ask / SELL bid).
DEAD     before the fill: price past fib 0 (new move), or the entry level already touched on M5 between the move
         end and the zone confirmation; 10 days max.
SL       S816 = 0.816 ("just past 0.786", his way) | S886 = 0.886 | S100 = 1.0 (move start). Only SL beyond entry.
TP       1R (his) | 1.5R | 2R.  Max hold 3 days. Costs $7/lot RT + $0.05 slip on stops + real Dukascopy spread.
JUDGE    pick on 2016-20 only, then read its 2021-26 (already seen data for other Fib 7 tests - treat with care).
Chalana: python src\research\gold_fib_grid.py run | report
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.research.gold_fib_big import m5_arrays  # noqa: E402

OUT = ROOT / "data/research/gold_fib_grid"
SETS = [("M15", 2.5), ("H1", 2.5), ("H1", 3.5)]
ENTRIES = {"E705": 0.705, "E745": 0.7455, "E786": 0.786}
SLS = {"S816": 0.816, "S886": 0.886, "S100": 1.0}
TPS = {"1R": 1.0, "1.5R": 1.5, "2R": 2.0}
COMM, SLIP, OZ, DAY = 7.0, 0.05, 100.0, 86_400_000


def zone_rows(z, hi, lo, op, ct):
    bull, H, L = bool(z["bull"]), z["H"], z["L"]
    lvl = lambda f: H - f * (H - L) if bull else L + f * (H - L)  # noqa: E731
    c_ms = int(z["created"].value // 10**6)
    tk = te.ticks("XAUUSD", c_ms, c_ms + 10 * DAY)
    if tk is None:
        return None
    t, a, b = tk
    i0 = np.searchsorted(op, int(z["t_end"].timestamp() * 1000), side="right")
    i1 = np.searchsorted(ct, c_ms, side="right")
    mh, ml = hi[i0:i1], lo[i0:i1]
    # running "fib 0 broken" on ticks (bid chart)
    broke = np.maximum.accumulate(b > H) if bull else np.maximum.accumulate(b < L)
    out = {}
    for ek, ef in ENTRIES.items():
        e = lvl(ef)
        pre = (ml <= e).any() if bull else (mh >= e).any()
        pre |= (mh > H).any() if bull else (ml < L).any()
        if pre:
            continue
        hit = np.nonzero(b <= e)[0] if bull else np.nonzero(b >= e)[0]
        if not len(hit):
            continue
        i = hit[0]
        if broke[i]:
            continue
        px = float(a[i]) if bull else float(b[i])
        j = np.searchsorted(t, t[i] + te.HOLD_MS)
        tt, aa, bb = t[i:j], a[i:j], b[i:j]
        if t[i] + te.HOLD_MS > t[-1]:                       # hold runs past the 10-day window -> reload
            tt, aa, bb = te.ticks("XAUUSD", int(t[i]), int(t[i]) + te.HOLD_MS)
        out[f"fill_{ek}"] = int(t[i])
        for sk, sf in SLS.items():
            if sf <= ef + 0.01:
                continue
            sl = lvl(sf)
            risk = (px - sl) if bull else (sl - px)
            if risk <= 0:
                continue
            for tk_, tr in TPS.items():
                res = te.exit_arrays(tt, aa, bb, bull, px, sl, "tp", tr)
                out[f"r_{ek}_{sk}_{tk_}"] = res["r"] - COMM / (risk * OZ) - (SLIP / risk if res["reason"] == "sl" else 0.0)
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    hi, lo, op, ct = m5_arrays()
    t_start = time.time()
    for tf, k in SETS:
        z = pd.read_parquet(ROOT / f"data/research/gold_fib_ml/{tf}_K{k}_zones_all.parquet")
        for y in sorted(z.created.dt.year.unique()):
            f = OUT / f"{tf}_K{k}_{y}.parquet"
            if f.exists():
                continue
            t0, rows = time.time(), []
            for r in z[z.created.dt.year == y].to_dict("records"):
                o = zone_rows(r, hi, lo, op, ct)
                if o:
                    rows.append({**{c: r[c] for c in ["tf", "bull", "created", "H", "L", "size", "atr"]}, **o})
            pd.DataFrame(rows).to_parquet(f)
            print(tf, k, y, len(rows), f"{time.time() - t0:.0f}s", flush=True)
            if time.time() - t_start > 120:
                return
    print("all done")


def pf(s):
    s = s.dropna()
    return round(float(s[s > 0].sum() / -s[s < 0].sum()), 2) if (s < 0).any() else np.nan


def report():
    rows = []
    for tf, k in SETS:
        d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob(f"{tf}_K{k}_20*.parquet"))], ignore_index=True)
        d["yr"] = d.created.dt.year
        for c in [c for c in d.columns if c.startswith("r_")]:
            a, b = d[d.yr <= 2020][c].dropna(), d[d.yr >= 2021][c].dropna()
            _, e, s, t = c.split("_")
            rows.append(dict(set=f"{tf} K{k}", entry=e, sl=s, tp=t, tr_yr_in=round(len(a) / 5), pf_16_20=pf(a), R_in=round(a.sum(), 1),
                             tr_yr_out=round(len(b) / 5.75), pf_21_26=pf(b), R_out=round(b.sum(), 1)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "grid_report.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    for s, g in r.groupby("set"):
        print(f"\n=== {s} ===")
        print(g.sort_values("pf_16_20", ascending=False).drop(columns="set").to_string(index=False))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
