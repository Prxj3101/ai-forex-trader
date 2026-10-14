r"""
ORB TP TEST (15 Oct 2026, his question: TP always 1:2 - would 1:1 or 1:5 be better?). Same 1,417 live-rule ORB trades
2019-26 as orb_be_test (walk-forward ML + compression + 1 per pair per day), replayed on ticks from the break to 16:00 NY.
Only the TP changes: 1R / 1.5R / 2R (current) / 2.5R / 3R / 4R / 5R / none (close 16:00 NY). SL = other side of the range.
Costs $5/lot + 0.2 pip on stop exits. Pass for a change = better PF AND total R than 2R in BOTH halves (2019-22, 2023-26).
Note: the ORB ML model was trained with the 2R outcome, which can favour 2R a little.
Chalana: python src\research\orb_tp_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import orb_be_test as be  # noqa: E402

be.VARIANTS = {f"TP{t}": (t, None, None) for t in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]} | {"EOD": (None, None, None)}

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    x = be.run()
    rows = []
    for v in be.VARIANTS:
        c = x[v]
        rows.append(dict(tp=v, pf=be.pf(c), win=round(100 * float((c > 0).mean()), 1), total_r=round(float(c.sum()), 1),
                         avg_r=round(float(c.mean()), 3), pf_19_22=be.pf(x[x.year <= 2022][v]), R_19_22=round(x[x.year <= 2022][v].sum(), 1),
                         pf_23_26=be.pf(x[x.year >= 2023][v]), R_23_26=round(x[x.year >= 2023][v].sum(), 1),
                         hit_tp=int((x[v + "_why"] == "tp").sum()), closed_16=int((x[v + "_why"] == "time").sum())))
    print("trades", len(x))
    print(pd.DataFrame(rows).to_string(index=False))
    x.to_parquet(ROOT / "data/research/orb_tp_test.parquet")
