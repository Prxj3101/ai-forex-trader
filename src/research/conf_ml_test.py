"""
Confluence ML test (16 Oct 2026, his "kr test"). Does ML trained FOR confluence beat the current setup?
RULES (fixed before the run). Data = zone_life/life5 (M5 labels) + h4_zone flag (confluence/{SYM}_conf.parquet).
Walk-forward 2019-26 (train only on earlier years, 5-day purge), bot rule = M15 touch 1 + VA + activity + exit brain.
  BASE : current 15m zone_brain (no h4 info), top 30%  -> confluence trades = h4_zone YES (PF 1.19 earlier)
  A    : same model + "h4_zone" as an extra feature, top 30% -> confluence / normal split
  B    : own model trained ONLY on confluence rows (M15 touch 1 + h4_zone), top 30 / 50 / 70% of its training scores;
         exit = the main exit brain's pick of the same trade
Pass: confluence PF better than BASE in BOTH 2019-21 and 2022-26, with >= 80% of BASE's trades, and (A) normal
trades not worse. Otherwise keep the live setup.
Run: python src/research/conf_ml_test.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml import walkforward as wf  # noqa: E402

BASE = ROOT / "data/research/zone_life/life5"
CONF = ROOT / "data/research/confluence"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
KEY = ["symbol", "zone_id", "touch_number"]


def pf(r):
    r = pd.Series(r, dtype=float)
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl > 0 else None


def row(name, g, col="r_ml"):
    return dict(model=name, trades=len(g), win=round(100 * float((g[col] > 0).mean()), 1) if len(g) else None,
                pf=pf(g[col]), pf_19_21=pf(g.loc[g.test_year <= 2021, col]), pf_22_26=pf(g.loc[g.test_year >= 2022, col]),
                total_r=round(float(g[col].sum()), 1))


def load():
    conf = pd.concat([pd.read_parquet(CONF / f"{s}_conf.parquet")[KEY + ["h4_zone"]] for s in SYMS], ignore_index=True)
    d = wf.load(SYMS, folder=BASE).merge(conf, on=KEY, how="left")
    for c in wf.CATEGORICAL:
        d[c] = d[c].astype("category")
    d["h4_zone"] = d["h4_zone"].astype("boolean").fillna(False).astype(float)
    return d


def bot(w):
    return w[(w.zone_tf == "M15") & (w.touch_number == 1) & w["keep_0.3"] & w.va_ok & w.act_ok]


def main():
    d = load()
    out = []
    base = wf.walk_forward(d, 2019, log=lambda *a: None)
    b0 = bot(base)
    out += [row("BASE confluence", b0[b0.h4_zone > 0]), row("BASE normal", b0[b0.h4_zone == 0])]

    feats = list(wf.FEATURES)
    wf.FEATURES[:] = feats + ["h4_zone"]
    try:
        wa = wf.walk_forward(d, 2019, log=lambda *a: None)
    finally:
        wf.FEATURES[:] = feats
    ba = bot(wa)
    out += [row("A (h4 as feature) confluence", ba[ba.h4_zone > 0]), row("A (h4 as feature) normal", ba[ba.h4_zone == 0])]

    # B: own model on confluence rows only
    cr = d[(d.zone_tf == "M15") & (d.touch_number == 1) & (d.h4_zone > 0)]
    exit_pick = base.set_index(KEY)[["r_ml"]]
    res = {k: [] for k in (0.3, 0.5, 0.7)}
    for year in range(2019, int(cr.test_year.max()) + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        tr = cr[cr.fill_time < start - pd.Timedelta(days=wf.PURGE_DAYS)]
        te_ = cr[cr.test_year == year].copy()
        if te_.empty or len(tr) < 300:
            continue
        m = lgb.LGBMClassifier(**dict(wf.PARAMS, min_child_samples=60)).fit(tr[feats], tr["y"])
        s_tr = m.predict_proba(tr[feats])[:, 1]
        te_["sc"] = m.predict_proba(te_[feats])[:, 1]
        for k in res:
            keep = te_[te_.sc >= np.quantile(s_tr, 1 - k)]
            res[k].append(keep)
    for k, parts in res.items():
        g = pd.concat(parts)
        g = g[g.va_ok & g.act_ok].join(exit_pick, on=KEY, rsuffix="_x")
        g["r_ml"] = g["r_ml"].fillna(g["r_tp1"])
        out.append(row(f"B (own model) confluence top {int(k * 100)}%", g))
    rep = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))
    rep.to_csv(CONF / "conf_ml_test.csv", index=False)


if __name__ == "__main__":
    main()
