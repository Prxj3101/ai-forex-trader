r"""
STEP 1 (9 Oct 2026): which EXIT for the whole current bot (wick entry, ML top 30% + VA + activity, M15 touch1)?
Same walk-forward entry selection (unseen 2019-2026), only the exit changes:
  always 1:1 | always 1:2 | always trail | exit brain (current: 1:1 vs trail) | exit brain 3-way (1:1 / 1:2 / trail)
Variant B: entry ML trained on target "1:2 wins" instead of "trail wins" + always 1:2.
Keep a change only if better in BOTH 2019-20.. (first half 2019-2022) and 2023-2026. 3 seeds for B.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan  # noqa: E731


def bot(sel):
    return sel[(sel.zone_tf.astype(str) == "M15") & (sel.touch_number == 1) & sel["keep_0.3"]
               & sel.va_ok.astype(bool) & sel.act_ok.astype(bool)]


def summary(name, r, yr, sym):
    a, b = r[yr <= 2022], r[yr >= 2023]
    return dict(variant=name, trades=len(r), win=round(100 * float((r > 0).mean()), 1), pf=pf(r), pf_2019_22=pf(a),
                pf_2023_26=pf(b), total_R=round(float(r.sum())), pos_years=int((r.groupby(yr).sum() > 0).sum()),
                pairs={s: pf(r[sym == s]) for s in sorted(sym.unique())})


def main():
    d = wf.load(config.CORE_SYMBOLS)
    out = []
    w = wf.walk_forward(d, log=lambda *a: None)          # entry ML exactly as the live bot (target trail win)
    # 3-way exit brain, trained per test year on training rows only
    picks = []
    for year in sorted(w.test_year.unique()):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr = d[d.fill_time < start - pd.Timedelta(days=wf.PURGE_DAYS)]
        te = w[w.test_year == year]
        p2 = lgb.LGBMRegressor(**wf.EXIT_PARAMS).fit(tr[wf.FEATURES], (tr.r_tp2 - tr.r_tp1).clip(-3, 3)).predict(te[wf.FEATURES])
        pt = lgb.LGBMRegressor(**wf.EXIT_PARAMS).fit(tr[wf.FEATURES], (tr.r_trail - tr.r_tp1).clip(-3, 3)).predict(te[wf.FEATURES])
        ch = np.argmax(np.vstack([np.zeros(len(te)), p2, pt]), axis=0)
        picks.append(pd.Series(np.choose(ch, [te.r_tp1.values, te.r_tp2.values, te.r_trail.values]), index=te.index))
    w["r_brain3"] = pd.concat(picks)
    s = bot(w)
    yr, sym = s.fill_time.dt.year, s.symbol.astype(str)
    for name, col in (("always 1:1", "r_tp1"), ("always 1:2", "r_tp2"), ("always trail", "r_trail"),
                      ("exit brain (CURRENT, 1:1 vs trail)", "r_ml"), ("exit brain 3-way (1:1/1:2/trail)", "r_brain3")):
        out.append(summary(name, s[col], yr, sym))
    # Variant B: entry ML target = 1:2 win
    for seed in (1, 2, 3):
        d2 = d.copy(); d2["y"] = (d2.r_tp2 > 0).astype(int)
        old = dict(wf.PARAMS); wf.PARAMS["random_state"] = seed
        w2 = wf.walk_forward(d2, log=lambda *a: None)
        wf.PARAMS.clear(); wf.PARAMS.update(old)
        s2 = bot(w2)
        out.append(summary(f"B: entry ML for 1:2 + always 1:2 (seed {seed})", s2.r_tp2, s2.fill_time.dt.year, s2.symbol.astype(str)))
    t = pd.DataFrame(out)
    t.to_csv(ROOT / "data/research/step1_exit.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
