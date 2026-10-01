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
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
RESEARCH = ROOT / "data" / "research"
MODELS = ROOT / "models"

# Sirf wo features jo trade lene se PEHLE pata hote hain.
# (track_zones ke first_touch_depth / touches / invalidated_time future hain -> kabhi use nahi karna)
NUMERIC = [
    "disp_atr", "bars_ob_to_bos", "disp_h1", "disp_h4", "break_margin_atr", "zone_height_atr",
    "pullback_atr", "spread_atr", "spread_r", "risk_atr", "range_pos_h4", "range_pos_d1",
    "bars_since_created", "touch_number", "fill_hour_ny", "fill_weekday", "hour_ny", "weekday",
    # Prajwal ke screenshots se (1 Oct 2026): displacement quality, local trend, approach momentum
    "impulse_candles", "impulse_body_pct", "disp_d1", "followthrough_atr",
    "trend_tf_aligned", "trend_h1_aligned", "approach_1h_atr", "approach_3h_atr", "approach_push",
    # Cost-aware (1 Oct 2026): is trade ka kharcha kitna R hai (commission / SL size) + SL pips
    "cost_r", "sl_pips",
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
EXIT_PARAMS = dict(          # exit model: same simple setup, lekin regression (R ka farak predict karta hai)
    objective="regression", learning_rate=0.03, n_estimators=300, num_leaves=15,
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, verbose=-1,
)
KEEP_TOP = [1.0, 0.7, 0.5, 0.3]     # 1.0 = saare trades (baseline), 0.5 = model ke top 50%
PURGE_DAYS = 5

# COST-AWARE: labels (r_trail, r_tp1, r_tp2) mein se commission + slippage KAAT dete hain,
# taaki ML "kharche ke BAAD jeeta ya nahi" seekhe. Gross values r_*_gross mein bachi rehti hain.
COMMISSION = 5.0      # USD per 1.00 lot round trip (Funding Pips USDJPY = $5; ECN brokers ~$6-7)
SLIP_PIPS = 0.2       # SL / trailing / breakeven exit pe itne pips extra nuksaan
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}


def add_costs(d: pd.DataFrame, commission: float, slip_pips: float) -> pd.DataFrame:
    pip = d["symbol"].astype(str).map(PIP)
    usd_quote = d["symbol"].astype(str).str.endswith("USD")
    per_lot = np.where(usd_quote, d["risk"] * 100_000, d["risk"] * 100_000 / d["entry"])   # SL pe 1 lot ka USD
    d["cost_r"] = commission / per_lot                   # commission R mein (pehle se pata hota hai)
    d["sl_pips"] = d["risk"] / pip
    slip_r = slip_pips * pip / d["risk"]
    for c in ("r_trail", "r_tp1", "r_tp2"):
        d[c + "_gross"] = d[c]
        stop_exit = d[c + "_exit"].isin(["sl", "trail", "be"])
        d[c] = d[c] - d["cost_r"] - np.where(stop_exit, slip_r, 0.0)
    return d


def load(symbols: list[str], commission: float = COMMISSION, slip_pips: float = SLIP_PIPS,
         folder: Path = RESEARCH) -> pd.DataFrame:
    parts = []
    for s in symbols:
        f = folder / f"{s}_v2.parquet"
        if not f.exists():
            raise SystemExit(f"{f.name} nahi mila. Pehle: python src\\research\\zone_dataset_v2.py --symbol {s}")
        parts.append(pd.read_parquet(f))
    d = pd.concat(parts, ignore_index=True)
    d["spread_r"] = d["spread_atr"] * d["atr"] / d["risk"]      # spread risk ka kitna hissa
    d["risk_atr"] = d["risk"] / d["atr"]                         # SL kitna bada (ATR mein)
    d = add_costs(d, commission, slip_pips)                      # ab r_trail / r_tp1 / r_tp2 = NET
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


def walk_forward(d: pd.DataFrame, first_test_year: int = 2019, log=print) -> pd.DataFrame:
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

        # --- SMART EXIT: har trade ke liye 1:1 ya trailing? (dono sirf TRAINING data se seekhte hain) ---
        # (a) Rule: price zone mein dheere aaya (approach neeche wale 1/3 mein) -> 1:1, warna trailing
        appr_thr = train["approach_1h_atr"].quantile(1 / 3)
        test["exit_rule"] = np.where(test["approach_1h_atr"] < appr_thr, "tp1", "trail")
        # (b) ML exit model: predict karta hai "trailing 1:1 se kitna R zyada dega". > 0 -> trailing
        ex = lgb.LGBMRegressor(**EXIT_PARAMS)
        ex.fit(train[FEATURES], (train["r_trail"] - train["r_tp1"]).clip(-3, 3))
        test["exit_ml"] = np.where(ex.predict(test[FEATURES]) > 0, "trail", "tp1")
        for m in ("rule", "ml"):
            test[f"r_{m}"] = np.where(test[f"exit_{m}"] == "trail", test["r_trail"], test["r_tp1"])
        out.append(test)
        log(f"  {year}: trained on {len(train):,} trades, tested on {len(test):,}")
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

    # Exit comparison: same trades (M15 pehla touch + ML top 30%), 4 alag exit tarike
    sel = wf[(wf["zone_tf"] == "M15") & (wf["touch_number"] == 1) & wf["keep_0.3"]].sort_values("fill_time")
    exits = []
    for col, label in (("r_trail", "Always trailing"), ("r_tp1", "Always 1:1"),
                       ("r_rule", "Smart rule (slow approach = 1:1)"), ("r_ml", "ML exit model")):
        cum = sel[col].cumsum()
        yearly = sel.groupby("test_year")[col].sum()
        exits.append(dict(exit=col.replace("r_", ""), label=label, trades=len(sel), pf=pf(sel[col]),
                          total_r=round(float(sel[col].sum()), 1), max_dd_r=round(float((cum.cummax() - cum).max()), 1),
                          losing_years=int((yearly < 0).sum())))

    # ASLI BOT = M15 pehla touch + ML top 30% + Exit Brain (sab NET of costs). Dashboard ka main number yahi.
    r = sel["r_ml"]
    cum = r.cumsum()
    bot = dict(
        label="M15 first touch + ML top 30% + exit brain",
        trades=int(len(sel)), win_pct=round(100 * float((r > 0).mean()), 1), pf=pf(r),
        total_r=round(float(r.sum()), 1), avg_r=round(float(r.mean()), 3),
        max_dd_r=round(float((cum.cummax() - cum).max()), 1),
        losing_years=int((sel.groupby("test_year")["r_ml"].sum() < 0).sum()),
        trail_share_pct=round(100 * float((sel["exit_ml"] == "trail").mean()), 1),
        by_year={int(y): round(float(g.sum()), 1) for y, g in sel.groupby("test_year")["r_ml"]},
        by_symbol={str(s): dict(trades=int(len(g)), pf=pf(g["r_ml"]), total_r=round(float(g["r_ml"].sum()), 1))
                   for s, g in sel.groupby("symbol", observed=True)},
    )
    return dict(summary=rows, by_year=by_year, by_symbol=by_sym, score_buckets=buckets, exits=exits, bot=bot)


SCORE_COLS = ["symbol", "zone_tf", "direction", "zone_candle_time", "created_at", "touch_number", "fill_time",
              "entry", "sl", "score", "keep_0.5", "keep_0.3", "r_trail", "r_trail_exit",
              "exit_ml", "r_ml", "r_tp1"]


def train(symbols: list[str], first_test_year: int = 2019, log=print,
          commission: float = COMMISSION, folder: Path = RESEARCH, save: bool = True) -> tuple[dict, pd.DataFrame]:
    """
    Poora kaam: load -> walk-forward -> results -> final model save. Dashboard bhi yahi call karta hai.
    save=False: sirf test (settings sweep ke liye), koi file / model overwrite nahi hota.
    """
    d = load(symbols, commission, folder=folder)
    log(f"Dataset: {len(d):,} trades, {len(FEATURES)} features, pairs: {', '.join(symbols)}, "
        f"costs ${commission:g}/lot + {SLIP_PIPS} pip slippage")
    wf = walk_forward(d, first_test_year, log=log)
    res = results(wf)
    if not save:
        return res, wf

    # Final model: saare data pe (live use ke liye) + feature importance
    model = lgb.LGBMClassifier(**PARAMS).fit(d[FEATURES], d["y"])
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    imp = (100 * imp / imp.sum()).round(1)

    MODELS.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS / "zone_brain.txt"))
    exit_model = lgb.LGBMRegressor(**EXIT_PARAMS).fit(d[FEATURES], (d["r_trail"] - d["r_tp1"]).clip(-3, 3))
    exit_model.booster_.save_model(str(MODELS / "exit_brain.txt"))
    res.update(symbols=symbols, features=FEATURES, importance=imp.to_dict(), n_trades=len(d),
               commission=commission, slippage_pips=SLIP_PIPS, results_are="net of costs",
               first_test_year=first_test_year, trained_at=pd.Timestamp.now(tz="UTC").isoformat())
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (RESEARCH / "ml_walkforward.json").write_text(json.dumps(res, indent=2, default=str))
    # Har test trade ka OUT-OF-SAMPLE score (dashboard chart pe dikhane ke liye; ye model ne train mein nahi dekha tha)
    sc = wf[SCORE_COLS].copy()
    sc["symbol"] = sc["symbol"].astype(str)
    sc["zone_tf"] = sc["zone_tf"].astype(str)
    sc["direction"] = sc["direction"].astype(str)
    sc.to_parquet(RESEARCH / "ml_scores.parquet")
    return res, wf


def main() -> None:
    ap = argparse.ArgumentParser(description="ML Brain walk-forward test")
    ap.add_argument("--symbols", nargs="+", default=config.CORE_SYMBOLS)
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--commission", type=float, default=COMMISSION,
                    help="USD per lot round trip (0 = purana gross mode, bina kharche)")
    a = ap.parse_args()

    res, _ = train(a.symbols, a.first_test_year, commission=a.commission)
    imp = pd.Series(res["importance"])

    pd.set_option("display.width", 200)
    print("\n== TEST RESULTS (sirf unseen saal) ==")
    print(pd.DataFrame(res["summary"]).to_string(index=False))
    print("\n== Score buckets (zyada score = zyada jeet?) ==")
    print(pd.DataFrame(res["score_buckets"]).to_string(index=False))
    print("\n== PF by year ==")
    print(pd.DataFrame(res["by_year"]).pivot(index="year", columns="keep", values="pf").to_string())
    print("\n== PF by pair ==")
    print(pd.DataFrame(res["by_symbol"]).pivot(index="symbol", columns="keep", values="pf").to_string())

    print("\n== Exit comparison: M15 touch 1 + ML top 30%, unseen years ==")
    print(pd.DataFrame(res["exits"]).drop(columns="exit").to_string(index=False))

    print("\n== Feature importance (model ne kis cheez pe sabse zyada dhyan diya, %) ==")
    print(imp.head(15).to_string())
    print("\nsaved -> models/zone_brain.txt, models/exit_brain.txt, data/research/ml_walkforward.json, data/research/ml_scores.parquet")


if __name__ == "__main__":
    main()
