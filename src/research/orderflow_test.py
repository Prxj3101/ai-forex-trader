r"""
ORDERFLOW TEST: kya asli CME orderflow bot ko behtar banata hai?

Data: CME futures trades 2024-01 -> 2026-09 (Databento). Usse pehle ke trades mein orderflow = NaN.

Do tareeke, dono ka FINAL TEST sirf 2025-01 -> 2026-09 pe (jo kisi ne seekha nahi):

  A) ML: LightGBM ko 46 normal features + 10 orderflow features. Walk-forward test saal 2025, 2026
     (2025 ka model 2024 tak seekhta hai -> orderflow wala hissa sirf 2024 se seekh paata hai).
     Compare: wahi test saal, bina orderflow ke.

  B) RULE (veto, Value Area filter jaisa): bot ke trades (2024) pe har feature ke 5 hisse (quintile).
     Sabse kharab hissa 2024 mein jo tha, uske threshold se 2025-26 mein woh trades skip -> PF badha?

Pehle: python src\features\orderflow.py          (raw trades -> 1-minute bars)
Phir:  python src\research\orderflow_test.py
Output: data/research/orderflow_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.features import orderflow as of  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402

RESEARCH = ROOT / "data" / "research"
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
OF_START = pd.Timestamp("2024-01-01", tz="UTC")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    gl = -r[r < 0].sum()
    return dict(trades=int(len(r)), win_pct=round(float((r > 0).mean() * 100), 1) if len(r) else None,
                pf=round(float(r[r > 0].sum() / gl), 3) if gl else None, total_r=round(float(r.sum()), 1))


def bot_sel(w: pd.DataFrame) -> pd.DataFrame:
    return w[(w["zone_tf"].astype(str) == "M15") & (w["touch_number"] == 1) & w["keep_0.3"] & w["va_ok"]]


def load_with_of(symbols, log=print) -> pd.DataFrame:
    d = wf.load(symbols)
    parts = []
    for s, g in d.groupby(d["symbol"].astype(str)):
        parts.append(of.add_orderflow_features(g, s))
    d = pd.concat(parts).sort_values("fill_time").reset_index(drop=True)
    cov = d.loc[d["fill_time"] >= OF_START, "of_delta_60m"].notna().mean()
    log(f"orderflow coverage on 2024+ trades: {cov:.1%}")
    return d


def run(symbols=None, log=print) -> dict:
    symbols = symbols or config.CORE_SYMBOLS
    d = load_with_of(symbols, log)
    res: dict = {"symbols": symbols, "test_period": "2025-01 .. latest", "results_are": "net of costs"}

    # ---- A) ML with vs without orderflow, test years 2025 + 2026 only ----
    base_feats = list(wf.FEATURES)
    out = {}
    for name, feats in (("without_orderflow", base_feats), ("with_orderflow", base_feats + of.OF_FEATURES)):
        wf.FEATURES = feats
        w = wf.walk_forward(d, first_test_year=2025, log=lambda *a: None)
        b = bot_sel(w)
        r = pd.Series(np.where(b["exit_ml"] == "trail", b["r_trail"], b["r_tp1"]), index=b.index)
        out[name] = dict(bot=stats(r), by_symbol={s: stats(r[b["symbol"].astype(str) == s]) for s in symbols},
                         by_year={int(y): stats(r[b["test_year"] == y]) for y in sorted(b["test_year"].unique())})
        log(f"ML {name}: {out[name]['bot']}")
        if name == "with_orderflow":
            import lightgbm as lgb
            train = d[d["fill_time"] < TEST_START]
            m = lgb.LGBMClassifier(**wf.PARAMS).fit(train[feats], train["y"])
            imp = pd.Series(m.booster_.feature_importance("gain"), index=feats)
            imp = (100 * imp / imp.sum()).sort_values(ascending=False)
            out[name]["orderflow_importance_pct"] = {k: round(float(v), 2) for k, v in imp[of.OF_FEATURES].items()}
            out[name]["orderflow_total_importance_pct"] = round(float(imp[of.OF_FEATURES].sum()), 1)
    wf.FEATURES = base_feats
    res["ml"] = out

    # ---- B) simple rules on the bot's trades (base ML) ----
    w = wf.walk_forward(d, first_test_year=2024, log=lambda *a: None)
    b = bot_sel(w).copy()
    b["r"] = np.where(b["exit_ml"] == "trail", b["r_trail"], b["r_tp1"])
    learn, test = b[b["fill_time"] < TEST_START], b[b["fill_time"] >= TEST_START]
    rules = {"baseline": dict(learn=stats(learn["r"]), test=stats(test["r"]))}
    for f in of.OF_FEATURES:
        q = learn[f].dropna()
        if len(q) < 200:
            continue
        edges = q.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
        bucket_pf = [stats(learn["r"][np.digitize(learn[f], edges) == k])["pf"] for k in range(5)]
        lo_bad = (bucket_pf[0] or 9) <= (bucket_pf[4] or 9)
        thr = edges[0] if lo_bad else edges[3]
        keep = (lambda x: ~(x[f] < thr)) if lo_bad else (lambda x: ~(x[f] > thr))   # NaN = keep
        rules[f] = dict(learn_quintile_pf=bucket_pf, drop="lowest 20%" if lo_bad else "highest 20%",
                        threshold=round(float(thr), 4),
                        learn=stats(learn.loc[keep(learn), "r"]),
                        test=stats(test.loc[keep(test), "r"]),
                        test_quintile_pf=[stats(test["r"][np.digitize(test[f], edges) == k])["pf"] for k in range(5)])
    res["rules"] = rules
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (RESEARCH / "orderflow_test.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = run()
    print("\n== A) ML, test 2025-2026 (bot trades, net) ==")
    for k, v in r["ml"].items():
        print(f"  {k:18s} {v['bot']}")
        print(f"  {'':18s} per pair: { {s: x['pf'] for s, x in v['by_symbol'].items()} }")
    print("  orderflow share of model importance:", r["ml"]["with_orderflow"].get("orderflow_total_importance_pct"), "%")
    print("\n== B) Rules: drop worst 20% (decided on 2024), checked on 2025-2026 ==")
    base = r["rules"]["baseline"]
    print(f"  baseline            learn {base['learn']['pf']}  test {base['test']['pf']} ({base['test']['trades']} trades)")
    for f, v in r["rules"].items():
        if f == "baseline":
            continue
        print(f"  {f:18s} drop {v['drop']:12s} learn {v['learn']['pf']}  test {v['test']['pf']} "
              f"({v['test']['trades']} trades)  test quintiles {v['test_quintile_pf']}")
    print("\nsaved -> data/research/orderflow_test.json")
