r"""
ML Brain v1: LightGBM + walk-forward test.

Sawaal: "Kya ML naye (unseen) data pe achhe trades ko bure trades se alag kar sakta hai?"

Kaise:
  1. Dataset v2 ke saare trades (EURUSD, USDJPY, GBPUSD) ek table mein.
  2. Target: win = trailing exit ka result > 0.
  3. WALK-FORWARD: har test saal Y ke liye model sirf Y se PEHLE ke saalon pe train hota hai,
     phir saal Y pe test. Jaise 2016-2018 pe train -> 2019 pe test, 2016-2019 pe train -> 2020 pe test...
     Test saal ka data model ne kabhi nahi dekha = asli live trading jaisa.
     PURGE: training ke aakhri 5 din hata dete hain (un trades ka result test saal mein khatam hota hai).
  4. Model har trade ko score deta hai (0-1). Threshold TRAINING data se decide hota hai
     (training predictions ka top X%), test data se nahi (warna cheating).
  5. Compare: saare trades vs ML ke chune hue trades (PF, total R, trades count).

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\ml\walkforward.py
    python src\ml\walkforward.py --symbols EURUSD USDJPY GBPUSD AUDUSD USDCAD

Output: terminal tables + data/research/ml_walkforward.json + models/zone_brain.txt (final model, saare data pe).
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "data" / "research"
MODELS = ROOT / "models"

# Sirf wo features jo trade lene se PEHLE pata hote hain.
# (track_zones ke first_touch_depth / touches / invalidated_time future hain -> kabhi use nahi karna)
NUMERIC = [
    "disp_atr", "bars_ob_to_bos", "disp_h1", "disp_h4", "break_margin_atr", "zone_height_atr",
    "pullback_atr", "spread_atr", "spread_r", "risk_atr", "range_pos_h4", "range_pos_d1",
    "bars_since_created", "touch_number", "fill_hour_ny", "fill_weekday", "hour_ny", "weekday",
]
BOOL = [
    "has_fvg", "h1_flow_aligned", "h1_confluence", "sweep_6h", "sweep_24h", "sweep_asia", "sweep_prev_day",
    "pd_aligned_h4", "pd_aligned_d1", "bias_h4_aligned", "bias_d1_aligned",
]
CATEGORICAL = ["symbol", "direction", "zone_tf", "killzone", "session_fill", "session_created"]
FEATURES = NUMERIC + BOOL + CATEGORICAL

PARAMS = dict(               # jaan-boojh ke "simple" model: overfitting se bachne ke liye
    objective="binary", learning_rate=0.03, n_estimators=300, num_leaves=15,
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, verbose=-1,
)
KEEP_TOP = [1.0, 0.7, 0.5, 0.3]     # 1.0 = saare trades (baseline), 0.5 = model ke top 50%
PURGE_DAYS = 5


def load(symbols: list[str]) -> pd.DataFrame:
    parts = []
    for s in symbols:
        f = RESEARCH / f"{s}_v2.parquet"
        if not f.exists():
            raise SystemExit(f"{f.name} nahi mila. Pehle: python src\\research\\zone_dataset_v2.py --symbol {s}")
        parts.append(pd.read_parquet(f))
    d = pd.concat(parts, ignore_index=True)
    d["spread_r"] = d["spread_atr"] * d["atr"] / d["risk"]      # spread risk ka kitna hissa
    d["risk_atr"] = d["risk"] / d["atr"]                         # SL kitna bada (ATR mein)
    d["test_year"] = d["fill_time"].dt.year
    d["y"] = (d["r_trail"] > 0).astype(int)
    for c in BOOL:
        d[c] = d[c].astype(float)
    for c in CATEGORICAL:
        d[c] = d[c].astype("category")
    return d.sort_values("fill_time").reset_index(drop=True)


def pf(r: pd.Series) -> float:
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def walk_forward(d: pd.DataFrame, first_test_year: int = 2019) -> pd.DataFrame:
    """Har test saal ke liye naya model. Returns test trades + score + keep flags."""
    out = []
    for year in range(first_test_year, int(d["test_year"].max()) + 1):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        train = d[d["fill_time"] < start - pd.Timedelta(days=PURGE_DAYS)]
        test = d[d["test_year"] == year].copy()
        if test.empty:
            continue
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(train[FEATURES], train["y"])
        train_score = model.predict_proba(train[FEATURES])[:, 1]
        test["score"] = model.predict_proba(test[FEATURES])[:, 1]
        for k in KEEP_TOP:
            thr = np.quantile(train_score, 1 - k) if k < 1 else -1.0   # threshold TRAINING se
            test[f"keep_{k}"] = test["score"] >= thr
        out.append(test)
        print(f"  {year}: trained on {len(train):,} trades, tested on {len(test):,}")
    return pd.concat(out, ignore_index=True)


def results(wf: pd.DataFrame) -> dict:
    rows, by_year, by_sym = [], [], []
    for k in KEEP_TOP:
        sel = wf[wf[f"keep_{k}"]]
        yearly = sel.groupby("test_year")["r_trail"].sum()
        rows.append(dict(keep=k, trades=len(sel), win_pct=round(100 * sel["y"].mean(), 1),
                         pf_trail=pf(sel["r_trail"]), pf_1to1=pf(sel["r_tp1"]), pf_1to2=pf(sel["r_tp2"]),
                         total_r=round(float(sel["r_trail"].sum()), 1), avg_r=round(float(sel["r_trail"].mean()), 3),
                         losing_years=int((yearly < 0).sum())))
        for y, g in sel.groupby("test_year"):
            by_year.append(dict(keep=k, year=int(y), trades=len(g), pf=pf(g["r_trail"]), total_r=round(float(g["r_trail"].sum()), 1)))
        for s, g in sel.groupby("symbol", observed=True):
            by_sym.append(dict(keep=k, symbol=s, trades=len(g), pf=pf(g["r_trail"]), total_r=round(float(g["r_trail"].sum()), 1)))
    # Score buckets: kya zyada score = sach mein zyada jeet? (model "calibrated" hai ya nahi)
    wf["bucket"] = pd.qcut(wf["score"], 5, labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "top 20%"])
    buckets = [dict(bucket=str(b), trades=len(g), avg_score=round(float(g["score"].mean()), 3),
                    real_win_pct=round(100 * g["y"].mean(), 1), pf=pf(g["r_trail"]))
               for b, g in wf.groupby("bucket", observed=True)]
    return dict(summary=rows, by_year=by_year, by_symbol=by_sym, score_buckets=buckets)


def main() -> None:
    ap = argparse.ArgumentParser(description="ML Brain walk-forward test")
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "USDJPY", "GBPUSD"])
    ap.add_argument("--first-test-year", type=int, default=2019)
    a = ap.parse_args()

    d = load(a.symbols)
    print(f"Dataset: {len(d):,} trades, {len(FEATURES)} features, pairs: {', '.join(a.symbols)}\n")
    print("Walk-forward:")
    wf = walk_forward(d, a.first_test_year)
    res = results(wf)

    pd.set_option("display.width", 200)
    print("\n== TEST RESULTS (sirf unseen saal) ==")
    print(pd.DataFrame(res["summary"]).to_string(index=False))
    print("\n== Score buckets (zyada score = zyada jeet?) ==")
    print(pd.DataFrame(res["score_buckets"]).to_string(index=False))
    print("\n== PF by year ==")
    print(pd.DataFrame(res["by_year"]).pivot(index="year", columns="keep", values="pf").to_string())
    print("\n== PF by pair ==")
    print(pd.DataFrame(res["by_symbol"]).pivot(index="symbol", columns="keep", values="pf").to_string())

    # Final model: saare data pe (live use ke liye) + feature importance
    model = lgb.LGBMClassifier(**PARAMS).fit(d[FEATURES], d["y"])
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    imp = (100 * imp / imp.sum()).round(1)
    print("\n== Feature importance (model ne kis cheez pe sabse zyada dhyan diya, %) ==")
    print(imp.head(15).to_string())

    MODELS.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS / "zone_brain.txt"))
    res.update(symbols=a.symbols, features=FEATURES, importance=imp.to_dict(), n_trades=len(d))
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (RESEARCH / "ml_walkforward.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nsaved -> models/zone_brain.txt, data/research/ml_walkforward.json")


if __name__ == "__main__":
    main()
