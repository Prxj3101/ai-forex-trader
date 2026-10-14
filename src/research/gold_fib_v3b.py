r"""V3b (14 Oct 2026): v3 results (2016-20) re-scored with the 2 bugs found while building round 3 (both = his own rules):
(a) zone dead if price went past fib 0 before the fill (new move), (b) 0.705 already touched on M5 before the fill.
No new run - filters the v3 parquet rows. FT2 (first-tap) fill only. Same pass rule: PF >= 1.10, >= 30 trades/yr."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
m5 = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"])
ms = lambda x: x.tz_convert(None).to_numpy().astype("datetime64[ms]").astype("int64")  # noqa: E731
hi, lo, op, ct = m5.high.to_numpy(), m5.low.to_numpy(), ms(m5.index), ms(pd.DatetimeIndex(m5.close_time))
pf = lambda s: round(float(s[s > 0].sum() / -s[s < 0].sum()), 3) if (s < 0).any() else np.nan  # noqa: E731
rows = []
for f in sorted((ROOT / "data/research/gold_fib_v3").glob("K*_*.parquet")):
    d = pd.read_parquet(f)
    d = d[d.fill_FT2.notna()].copy()
    ok = []
    for r in d.itertuples():
        i0 = np.searchsorted(op, int(r.t_end.timestamp() * 1000), side="right")
        i1 = np.searchsorted(ct, int(r.fill_FT2), side="right")
        h, l = hi[i0:i1], lo[i0:i1]
        bad = ((l <= r.entry).any() or (h > r.H).any()) if r.bull else ((h >= r.entry).any() or (l < r.L).any())
        ok.append(not bad)
    d["ok"] = ok
    yrs = d.created.dt.year.nunique()
    k, tf = f.stem.split("_")
    for c in ["r_FT2_B_1R", "r_FT2_B_1.5R", "r_FT2_C_1R", "r_FT2_C_1.5R"]:
        g = d[d.ok][c].dropna()
        rows.append(dict(K=k[1:], tf=tf, sl_tp=c[6:], kept=f"{int(d.ok.sum())}/{len(d)}", trades_yr=round(len(g) / yrs),
                         win=round(float((g > 0).mean()) * 100), pf_clean=pf(g), pf_before=pf(d[c].dropna()), R=round(g.sum(), 1)))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
