r"""
"EVERYTHING TOGETHER" walk-forward on CORRECT (tick) labels, 2016-2026.
Per group (M15 touch1 / H1 touch1) its own LightGBM: all normal features + HTF level features
(PDH/PDL, PWH/PWL, round numbers, same-side flags) ; target = net win for that exit (1:1 / 1:2 / trail);
expanding walk-forward, test years 2019-2026, threshold = quantile of TRAINING scores (top 30 / 20 / 10%), 3 seeds.
PASS (fixed before) = net PF >= 1.10 on unseen 2019-26, >= 6 of 8 years positive, >= 2 of 3 pairs > 1.
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
LVL = ["d_pdl", "d_pdh", "d_pwl", "d_pwh", "d_round", "at_pd_side", "at_pw_side"]


def levels(d):
    out = []
    for sym, g in d.groupby(d.symbol.astype(str)):
        D = pd.read_parquet(ROOT / f"data/processed/{sym}_D1.parquet"); W = pd.read_parquet(ROOT / f"data/processed/{sym}_W1.parquet")
        ctd = pd.DatetimeIndex(D["close_time"]).values; ctw = pd.DatetimeIndex(W["close_time"]).values
        g = g.copy()
        i = np.searchsorted(ctd, g.created_at.values, side="right") - 1
        j = np.searchsorted(ctw, g.created_at.values, side="right") - 1
        ref = np.where(g.direction.astype(str) == "bull", g.bottom, g.top)
        for name, arr in (("pdl", D.low.values[i]), ("pdh", D.high.values[i]), ("pwl", W.low.values[j]), ("pwh", W.high.values[j])):
            g[f"d_{name}"] = (ref - arr) / g.atr
        step = 0.0050 if sym != "USDJPY" else 0.50
        g["d_round"] = np.abs(ref - np.round(ref / step) * step) / g.atr
        bull = g.direction.astype(str) == "bull"
        g["at_pd_side"] = np.where(bull, g.d_pdl.abs() <= 1, g.d_pdh.abs() <= 1).astype(float)
        g["at_pw_side"] = np.where(bull, g.d_pwl.abs() <= 1, g.d_pwh.abs() <= 1).astype(float)
        out.append(g)
    return pd.concat(out)


def main():
    d = levels(wf.load(config.CORE_SYMBOLS))
    feats = wf.FEATURES + LVL
    rows = []
    for tf in ("M15", "H1"):
        g = d[(d.zone_tf.astype(str) == tf) & (d.touch_number == 1)].sort_values("fill_time")
        for ex in ("r_tp1", "r_tp2", "r_trail"):
            res = {k: [] for k in (0.3, 0.2, 0.1)}
            for seed in (1, 2, 3):
                for year in range(2019, 2027):
                    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
                    tr = g[g.fill_time < start - pd.Timedelta(days=5)]; te = g[g.fill_time.dt.year == year]
                    m = lgb.LGBMClassifier(**dict(wf.PARAMS, random_state=seed)).fit(tr[feats], (tr[ex] > 0).astype(int))
                    s_tr = m.predict_proba(tr[feats])[:, 1]; s_te = m.predict_proba(te[feats])[:, 1]
                    for k in res:
                        sel = te[s_te >= np.quantile(s_tr, 1 - k)]
                        res[k].append(sel.assign(seed=seed, r=sel[ex]))
            for k, parts in res.items():
                x = pd.concat(parts)
                yrs = x.groupby(x.fill_time.dt.year)["r"].sum() / 3
                pairs = [pf(q.r) for _, q in x.groupby(x.symbol.astype(str))]
                ok = pf(x.r) >= 1.10 and (yrs > 0).sum() >= 6 and sum(p > 1 for p in pairs) >= 2
                rows.append(dict(tf=tf, exit=ex, top=k, trades_per_seed=len(x) // 3, win=round(100 * (x.r > 0).mean(), 1),
                                 net_pf=pf(x.r), positive_years=int((yrs > 0).sum()), pairs=pairs, passed=ok))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows); t.to_csv(ROOT / "data/research/all_in_wf.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
