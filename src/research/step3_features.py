r"""
STEP 3 (9 Oct 2026): new ML features for the current bot (wick entry, M15 touch1, ML top 30% + VA + activity,
exit brain). Added to the ML (not as hard filters):
  naked POC (vp_naked_r, vp_naked_ok), weekly / daily level distance (d_pwl, d_pwh, d_pdl, d_pdh, at_pw_side,
  at_pd_side), session open vs previous value area (vp_open_side), previous session / day VA (vp_sess_edge,
  vp_day_edge, vp_day_pos).
Baseline vs +features, 3 seeds each (LightGBM randomness). Keep only if better in 2019-22 AND 2023-26 on average
and in the account simulation.
Chalana: python src\research\step3_features.py build | python src\research\step3_features.py run base | run new
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402
CACHE = ROOT / "data/research/step3_data.parquet"
NEW = ["vp_naked_r", "vp_naked_ok", "d_pwl", "d_pwh", "d_pdl", "d_pdh", "at_pw_side", "at_pd_side",
       "vp_open_side", "vp_sess_edge", "vp_day_edge", "vp_day_pos"]
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan  # noqa: E731


def build():
    from src.research.session_vp_test import features
    from src.research.all_in_wf import levels
    d = levels(features(wf.load(config.CORE_SYMBOLS)))
    d.to_parquet(CACHE)
    print("rows", len(d), d[NEW].notna().mean().round(2).to_dict())


def run(kind):
    d = pd.read_parquet(CACHE)
    for c in wf.CATEGORICAL:
        d[c] = d[c].astype("category")
    feats = wf.FEATURES + (NEW if kind == "new" else [])
    orig = list(wf.FEATURES)
    rows = []
    for seed in (1, 2, 3):
        wf.FEATURES[:] = feats
        p0 = dict(wf.PARAMS); wf.PARAMS["random_state"] = seed
        e0 = dict(wf.EXIT_PARAMS); wf.EXIT_PARAMS["random_state"] = seed
        w = wf.walk_forward(d, log=lambda *a: None)
        wf.FEATURES[:] = orig; wf.PARAMS.clear(); wf.PARAMS.update(p0); wf.EXIT_PARAMS.clear(); wf.EXIT_PARAMS.update(e0)
        s = w[(w.zone_tf.astype(str) == "M15") & (w.touch_number == 1) & w["keep_0.3"] & w.va_ok.astype(bool) & w.act_ok.astype(bool)]
        r, yr = s.r_ml, s.fill_time.dt.year
        rows.append(dict(kind=kind, seed=seed, trades=len(s), win=round(100 * float((r > 0).mean()), 1), pf=pf(r),
                         pf_2019_22=pf(r[yr <= 2022]), pf_2023_26=pf(r[yr >= 2023]), total_R=round(float(r.sum())),
                         pos_years=int((r.groupby(yr).sum() > 0).sum())))
        if seed == 1:
            w.to_parquet(ROOT / f"data/research/step3_wf_{kind}.parquet")
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / f"data/research/step3_{kind}.csv", index=False)


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else run(sys.argv[2])
