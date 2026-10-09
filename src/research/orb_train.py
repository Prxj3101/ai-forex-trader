r"""
ORB BRAIN: final LightGBM model for the live ORB shadow (9 Oct 2026).

Same model / features / settings as the walk-forward test (src/research/orb_ml_test.py), trained once on ALL
ORB breakouts 2016-2026 (EUR/GBP/USDJPY, ASIA / LON30 / LON60), label = net R at 2R > 0.
Saves:
  models/orb_brain.txt         LightGBM model
  models/orb_meta.json         threshold (keep top 30% = 70th pct of training scores), tight cutoffs per range
                               (30th pct of w_ratio 2016-20), feature list, categories, settings
  data/research/orb_backtest_stats.json   honest walk-forward numbers (2019-26 test years, max 1 trade per pair
                               per day) by range x pair / year, for the dashboard ORB stats page
Chalana: python src\research\orb_train.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import orb_ml_test as om, orb_nr_test as nr  # noqa: E402

MODEL = ROOT / "models/orb_brain.txt"
META = ROOT / "models/orb_meta.json"
STATS = ROOT / "data/research/orb_backtest_stats.json"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
RANGES = ["ASIA", "LON30", "LON60"]
KEEP = 0.3
TP_R = 2.0
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def X_of(d):
    X = d[om.FEATURES].copy()
    X["symbol"] = pd.Categorical(X.symbol, categories=SYMBOLS)
    X["range"] = pd.Categorical(X["range"], categories=RANGES)
    return X


def tight_cutoffs():
    z = nr.load()
    return {r: float(z[(z.when.dt.year <= 2020) & (z.range == r)].w_ratio.quantile(0.3)) for r in RANGES}


def train(seed: int = 1):
    import lightgbm as lgb
    d = pd.read_parquet(om.FEAT)
    y = d["BRK_2R_net"]
    ok = y.notna()
    m = lgb.LGBMClassifier(random_state=seed, **PARAMS)
    m.fit(X_of(d[ok]), (y[ok] > 0).astype(int))
    s = m.predict_proba(X_of(d[ok]))[:, 1]
    thr = float(np.quantile(s, 1 - KEEP))
    MODEL.parent.mkdir(exist_ok=True)
    m.booster_.save_model(str(MODEL))
    meta = dict(trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), rows=int(ok.sum()),
                data_to=str(d.when.max().date()), threshold=round(thr, 5), keep=KEEP, tp_r=TP_R,
                tight_q30=tight_cutoffs(), features=om.FEATURES, symbols=SYMBOLS, ranges=RANGES, params=PARAMS,
                seed=seed, label="net R at 2R > 0 ($5/lot + 0.2 pip stop slippage)")
    META.write_text(json.dumps(meta, indent=2))
    print(f"ORB brain saved: {ok.sum()} rows, threshold {thr:.4f} (keep top {int(KEEP*100)}%)")
    return meta


def backtest_stats():
    """Walk-forward (out-of-sample) trades 2019-26, seed 1, ML keep 30% + (NR4 or TIGHT), max 1 per pair per day."""
    d, res = om.wf(seeds=(1,))
    q = tight_cutoffs()
    tight = d.apply(lambda r: r.w_ratio <= q[r["range"]], axis=1)
    x = d[res[("2R", 1, KEEP)] & ((d.nr4 == 1) | tight) & (d.year >= 2019)]
    x = x.sort_values("break_ms").groupby(["symbol", "day"]).head(1)
    c = "BRK_2R_net"
    pf = om.pf

    def summ(g):
        r = g[c]
        return dict(n=int(len(r)), pf=pf(r), win=round(100 * float((r > 0).mean()), 1), r=round(float(r.sum()), 1))
    eq = x[c].cumsum().to_numpy()
    out = dict(source="walk-forward test years 2019-2026 (model never saw the year it traded), seed 1, "
                      "ML top 30% + (NR4 or TIGHT), TP 2R, max 1 trade per pair per day, net of costs",
               total=summ(x), max_dd_r=round(float((np.maximum.accumulate(eq) - eq).max()), 1),
               by_range_pair=[dict(range=rg, symbol=s, **summ(g)) for (rg, s), g in x.groupby(["range", "symbol"])],
               by_range=[dict(range=rg, **summ(g)) for rg, g in x.groupby("range")],
               by_symbol=[dict(symbol=s, **summ(g)) for s, g in x.groupby("symbol")],
               by_year=[dict(year=int(yy), **summ(g)) for yy, g in x.groupby("year")],
               equity=[[int(ms // 1000), round(float(v), 2)] for ms, v in zip(x.break_ms, eq)])
    STATS.write_text(json.dumps(out))
    print("stats:", out["total"], "maxDD", out["max_dd_r"])


if __name__ == "__main__":
    train()
    backtest_stats()
