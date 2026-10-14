r"""
GOLD FIB 7 - BIG MOVES ONLY (15 Oct 2026). From his 9 TradingView examples: every move he marks is a BIG one-session
move (~0.9-3% of price, ~4-10 x M15 ATR). In his 200 M15 labels, size/ATR >= 6 was the only good bucket.
RULES FIXED BEFORE RUNNING (no peeking at 2021-26 until the config is frozen):
  MOVES   M15, ATR zigzag K = 2.5 (as round 3), fresh zone, H4 EMA50 trend (zones_v2), leg size >= 6 x ATR(14, M15).
  ENTRY   limit at 0.705, FIRST chart touch only (FT2: fill at that tick's ask for BUY / bid for SELL).
  DEAD    if price went past fib 0 before the fill (new move) or 0.705 was already touched on M5 (inner-bar tap),
          or 10 days old.
  EXIT    PRIMARY: SL 1.0 (move start), TP 1:1 (his way). Neighbours: size >= 5 / 8 ATR, SL 0.886, TP 1.5R.
  COSTS   $7/lot RT + $0.05 slip on stops + real Dukascopy spread.
  PASS    2016-20: PF >= 1.10, >= 20 trades/yr (big moves are rare), neighbours not far off.
          2021-26: run ONCE for the frozen primary config (he asked for it), PF >= 1.10 needed there too.
Chalana: python src\research\gold_fib_big.py run       (all years, resumable per year)
         python src\research\gold_fib_big.py report
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import gold_fib as gf  # noqa: E402
from src.research import gold_fib_v3 as v3  # noqa: E402

OUT = ROOT / "data/research/gold_fib_big"
MIN_ATR = 5.0           # store >= 5 so the 5 / 6 / 8 neighbours can be read from one run
K = 2.5


def m5_arrays():
    m5 = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"])
    ms = lambda x: x.tz_convert(None).to_numpy().astype("datetime64[ms]").astype("int64")  # noqa: E731
    return m5.high.to_numpy(), m5.low.to_numpy(), ms(m5.index), ms(pd.DatetimeIndex(m5.close_time))


def clean_flag(r, hi, lo, op, ct):
    i0 = np.searchsorted(op, int(r["t_end"].timestamp() * 1000), side="right")
    i1 = np.searchsorted(ct, int(r["fill_FT2"]), side="right")
    h, l = hi[i0:i1], lo[i0:i1]
    if r["bull"]:
        return not ((l <= r["entry"]).any() or (h > r["H"]).any())
    return not ((h >= r["entry"]).any() or (l < r["L"]).any())


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    zf = OUT / "zones_M15_K2.5.parquet"
    if zf.exists():
        z = pd.read_parquet(zf)
    else:
        m5 = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"])
        sess = gf.session_table(m5)
        z = v3.zones_k(K, "M15", gf.bars("M15"), sess, gf.bars("H4"))
        z = z[z["size"] >= MIN_ATR * z.atr].reset_index(drop=True)
        z.to_parquet(zf)
        print("zones >= 5 ATR:", len(z), flush=True)
    hi, lo, op, ct = m5_arrays()
    for y in sorted(z.created.dt.year.unique()):
        f = OUT / f"M15_{y}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        rows = []
        for _, r in z[z.created.dt.year == y].iterrows():
            res = {**r.to_dict(), **v3.trade(r)}
            res["clean"] = bool(res.get("fill_FT2") is not None and not pd.isna(res.get("fill_FT2")) and clean_flag(res, hi, lo, op, ct))
            rows.append(res)
        pd.DataFrame(rows).to_parquet(f)
        print(y, len(rows), f"{time.time() - t0:.0f}s", flush=True)


def pf(s):
    s = s.dropna()
    return round(float(s[s > 0].sum() / -s[s < 0].sum()), 2) if (s < 0).any() else np.nan


def report():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("M15_20*.parquet"))], ignore_index=True)
    d = d[d.clean].copy()
    d["satr"] = d["size"] / d.atr
    d["yr"] = pd.to_datetime(d.fill_FT2, unit="ms").dt.year
    rows = []
    for thr in [5, 6, 8]:
        g = d[d.satr >= thr]
        for c in ["r_FT2_C_1R", "r_FT2_C_1.5R", "r_FT2_B_1R", "r_FT2_B_1.5R"]:
            a, b = g[g.yr <= 2020][c], g[g.yr >= 2021][c]
            rows.append(dict(min_atr=thr, exit=c[6:], tr_yr_16_20=round(a.notna().sum() / 5), pf_16_20=pf(a), R_16_20=round(a.sum(), 1),
                             tr_yr_21_26=round(b.notna().sum() / 5.75), pf_21_26=pf(b), R_21_26=round(b.sum(), 1)))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    g = d[d.satr >= 6]
    print("\nPRIMARY (>=6 ATR, SL 1.0, TP 1:1) per year:")
    print(g.groupby("yr")["r_FT2_C_1R"].agg(trades="count", R="sum", pf=pf).round(2).to_string())


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
