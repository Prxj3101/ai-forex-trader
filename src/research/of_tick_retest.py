r"""
ORDERFLOW RETEST on CORRECT (tick) labels. Real CME futures orderflow (Databento 6E/6B/6J, 2024-01..2026-09).
Features (computed before the fill, unchanged): src/features/orderflow.py + src/research/orderflow_test2.py
(cached in data/research/orderflow_test2_data.parquet). Labels: tick-labelled v2 (net of $5 + 0.2 pip).

Rules fixed BEFORE looking:
  learn = 2024 trades, test = 2025-01 .. 2026-09 (never seen when choosing).
  (A) one feature at a time: in 2024 keep the better side (top 40% or bottom 40% of that feature),
      threshold from 2024 only -> apply to 2025-26.
  (B) LightGBM with OF + normal features, trained on 2024, top 30% (training-score cut) -> 2025-26;
      and trained on 2024+2025 -> 2026. 5 seeds averaged.
  PASS = test net PF >= 1.10, 2025 and 2026 both > 1, >= 100 trades, >= 2 of 3 pairs > 1.
Groups: M15 touch1, H1 touch1. Exits: 1:1, 1:2, trail (each reported).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402

KEY = ["symbol", "zone_tf", "zone_id", "touch_number", "fill_time"]
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 2) if (r < 0).any() else np.nan  # noqa: E731


def data():
    d = wf.load(config.CORE_SYMBOLS)
    of = pd.read_parquet(ROOT / "data/research/orderflow_test2_data.parquet")
    feats = [c for c in of.columns if c.startswith("of_")]
    of = of[KEY + feats].copy()
    for c in ("symbol", "zone_tf"):
        of[c] = of[c].astype(str); d[c] = d[c].astype(str)
    d = d.merge(of.drop_duplicates(KEY), on=KEY, how="left")
    d = d[(d.fill_time >= "2024-01-01")].copy()
    d = d[d[feats].notna().sum(axis=1) >= len(feats) // 2]
    return d, feats


def verdict(x):
    if len(x) < 100:
        return False
    y = x.fill_time.dt.year
    pairs = [pf(g["r"]) > 1 for _, g in x.groupby("symbol")]
    return pf(x["r"]) >= 1.10 and pf(x.loc[y == 2025, "r"]) > 1 and pf(x.loc[y == 2026, "r"]) > 1 and sum(pairs) >= 2


def main():
    d, feats = data()
    print("rows with orderflow:", len(d), "| features:", len(feats))
    out = []
    for tf in ("M15", "H1"):
        g0 = d[(d.zone_tf == tf) & (d.touch_number == 1)]
        learn, test = g0[g0.fill_time < "2025-01-01"], g0[g0.fill_time >= "2025-01-01"]
        for ex in ("r_tp1", "r_tp2", "r_trail"):
            base = test.assign(r=test[ex])
            out.append(dict(tf=tf, exit=ex, test="no filter", n=len(base), pf=pf(base.r), passed=verdict(base)))
            for f in feats:
                lv = learn[f].dropna()
                if len(lv) < 200:
                    continue
                lo, hi = lv.quantile(0.4), lv.quantile(0.6)
                pf_lo, pf_hi = pf(learn.loc[learn[f] <= lo, ex]), pf(learn.loc[learn[f] >= hi, ex])
                m = (test[f] <= lo) if pf_lo >= pf_hi else (test[f] >= hi)
                x = test[m].assign(r=test.loc[m, ex])
                out.append(dict(tf=tf, exit=ex, test=f"{f} {'low' if pf_lo >= pf_hi else 'high'}",
                                learn_pf=max(pf_lo, pf_hi), n=len(x), pf=pf(x.r),
                                pf_2025=pf(x.loc[x.fill_time.dt.year == 2025, "r"]),
                                pf_2026=pf(x.loc[x.fill_time.dt.year == 2026, "r"]), passed=verdict(x)))
            # (B) ML
            import lightgbm as lgb
            cols = wf.FEATURES + feats

            def X(df):
                x = df[cols].copy()
                for c in ("symbol", "zone_tf"):
                    if c in x:
                        x[c] = pd.Categorical(x[c].astype(str), categories=sorted(d[c].astype(str).unique()))
                return x
            for tr_end, te_start, te_end in (("2025-01-01", "2025-01-01", "2026-12-31"), ("2026-01-01", "2026-01-01", "2026-12-31")):
                tr = g0[g0.fill_time < tr_end]; te_ = g0[(g0.fill_time >= te_start) & (g0.fill_time <= te_end)]
                picks = []
                for seed in range(1, 6):
                    p = dict(wf.PARAMS, random_state=seed)
                    mdl = lgb.LGBMClassifier(**p).fit(X(tr), (tr[ex] > 0).astype(int))
                    thr = np.quantile(mdl.predict_proba(X(tr))[:, 1], 0.7)
                    sel = te_[mdl.predict_proba(X(te_))[:, 1] >= thr]
                    picks.append((len(sel), pf(sel[ex])))
                out.append(dict(tf=tf, exit=ex, test=f"ML(base+OF) train<{tr_end[:4]}", n=int(np.mean([a for a, _ in picks])),
                                pf=round(float(np.nanmean([b for _, b in picks])), 2), passed=None))
    t = pd.DataFrame(out)
    t.to_csv(ROOT / "data/research/of_tick_retest.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.sort_values(["tf", "exit", "pf"], ascending=[True, True, False]).groupby(["tf", "exit"]).head(6).to_string(index=False))
    print("\nPASSED:", t[t.passed == True][["tf", "exit", "test", "n", "pf"]].to_string(index=False) if (t.passed == True).any() else "none")  # noqa: E712


if __name__ == "__main__":
    main()
