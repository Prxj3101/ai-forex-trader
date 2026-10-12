r"""
ORB RESEARCH DATA for the dashboard (Research -> ORB): one row per day x pair x range (2016-2026, EUR/GBP/USDJPY,
ASIA / LON30 / LON60) with range, compression flags, breakout, results (1R / 2R / EOD, gross + net) and the
walk-forward ML score (2019-26: model trained only on earlier years; same settings as orb_ml_test / orb_train).
  status  traded      = compression + ML top 30% + first such break of the pair that day (= live bot rule)
          skip_daily  = would trade but the pair already had an ORB trade that day
          skip_ml     = compression ok, ML below threshold
          skip_nocomp = no compression (NR4 / tight)
          no_break    = compression ok, no break before 11:00 NY
          no_ml       = compression ok, broke, but year < 2019 (no out-of-sample ML score)
Output: data/research/orb_research.parquet
Chalana: python src\research\orb_research_data.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.live.orb import windows  # noqa: E402
from src.research import orb_ml_test as om, orb_nr_test as nr  # noqa: E402
from src.research.orb_train import KEEP, PARAMS, X_of, tight_cutoffs  # noqa: E402

OUT = ROOT / "data/research/orb_research.parquet"


def wf_scores(seed: int = 1) -> pd.DataFrame:
    import lightgbm as lgb
    d = pd.read_parquet(om.FEAT)
    d["year"] = d.when.dt.year
    y = d["BRK_2R_net"]
    d["score"], d["thr"] = np.nan, np.nan
    for Y in range(2019, 2027):
        tr, te_ = (d.year < Y) & y.notna(), d.year == Y
        m = lgb.LGBMClassifier(random_state=seed, **PARAMS)
        m.fit(X_of(d[tr]), (y[tr] > 0).astype(int))
        thr = float(np.quantile(m.predict_proba(X_of(d[tr]))[:, 1], 1 - KEEP))
        d.loc[te_, "score"] = m.predict_proba(X_of(d[te_]))[:, 1]
        d.loc[te_, "thr"] = thr
    d["day"] = d["day"].astype(str)
    return d[["symbol", "day", "range", "score", "thr"] + [f for f in om.FEATURES if f not in ("symbol", "range", "nr4", "w_ratio", "inside", "nr7")]]


def build():
    z = nr.load()
    z["day"] = z["day"].astype(str)
    q = tight_cutoffs()
    z["tight"] = (z.w_ratio <= z["range"].map(q)).astype(int)
    z["comp"] = ((z.nr4 == 1) | (z.tight == 1)).astype(int)
    z = z.merge(wf_scores(), on=["symbol", "day", "range"], how="left")
    z["year"] = pd.to_datetime(z.day).dt.year
    z["ml_keep"] = (z.score >= z.thr)
    cand = (z.broke == 1) & (z.comp == 1) & z.ml_keep & z.BRK_risk.notna()
    z["status"] = np.where(z.comp == 0, "skip_nocomp",
                  np.where(z.broke == 0, "no_break",
                  np.where(z.score.isna(), "no_ml",
                  np.where(~z.ml_keep, "skip_ml", "cand"))))
    first = z[cand].sort_values("break_ms").groupby(["symbol", "day"]).head(1).index
    z.loc[cand, "status"] = "skip_daily"
    z.loc[first, "status"] = "traded"
    w = [windows(d, r) for d, r in zip(z.day, z["range"])]
    z["t0"], z["t1"], z["w_end"], z["eod"] = (np.array([x[i] for x in w]) // 1000 for i in range(4))
    keep = ["symbol", "day", "year", "range", "status", "t0", "t1", "w_end", "eod", "hi", "lo", "w", "w_ratio", "nr4", "nr7",
            "inside", "tight", "comp", "broke", "direction", "break_ms", "BRK_entry", "BRK_risk",
            "BRK_1R", "BRK_1R_why", "BRK_2R", "BRK_2R_why", "BRK_EOD", "BRK_EOD_why", "BRK_1R_net", "BRK_2R_net", "BRK_EOD_net",
            "score", "thr", "ml_keep", "range_atr", "act_15m", "h4_aligned", "d1_mom_aligned", "prevday_aligned", "weekday"]
    z = z[[c for c in keep if c in z.columns]].rename(columns={"BRK_entry": "entry", "BRK_risk": "risk"})
    z.to_parquet(OUT)
    print(len(z), "rows ->", OUT.name)
    print(z.status.value_counts().to_dict())
    t = z[(z.status == "traded")]
    print("traded 2019-26:", len(t), "PF 2R net", om.pf(t.BRK_2R_net.dropna()))


if __name__ == "__main__":
    build()
