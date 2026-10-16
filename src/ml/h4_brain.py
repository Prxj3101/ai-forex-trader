"""
H4 BRAIN (16 Oct 2026): ML model ONLY for scoring H4 zones (zone_tf "H4") in the live zone bot.
The 15m bot's zone_brain / exit_brain are NOT touched.

Training data = data/research/zone_h4/n5/{SYM}_v2.parquet (M15 + H1 + H4 zones, H4 = BOS swing 5, life 60 d;
built by src/research/h4_merge_test.py build). Same features + params as walkforward.py (zone_tf "H4" category).
Cut-off = top 30% of the TRAINING scores (same rule as walk-forward).
Out-of-sample result (h4_merge_test report --n 5): H4 touch 1 + top 30% + VA + activity, 1:1 = PF 1.078.

Output: models/h4_brain.txt + models/h4_brain.json (threshold, features, trained_at, settings).
Run: python src/ml/h4_brain.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml import walkforward as wf  # noqa: E402

FOLDER = ROOT / "data" / "research" / "zone_h4" / "n5"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
H4_SWING, H4_LIFE_DAYS, KEEP = 5, 60, 0.3


def main():
    d = wf.load(SYMS, folder=FOLDER)
    model = lgb.LGBMClassifier(**wf.PARAMS).fit(d[wf.FEATURES], d["y"])
    sc = model.predict_proba(d[wf.FEATURES])[:, 1]
    thr = float(np.quantile(sc, 1 - KEEP))
    h4 = d["zone_tf"].astype(str) == "H4"
    wf.MODELS.mkdir(exist_ok=True)
    model.booster_.save_model(str(wf.MODELS / "h4_brain.txt"))
    meta = dict(threshold=thr, keep=KEEP, features=wf.FEATURES, h4_swing=H4_SWING, h4_life_days=H4_LIFE_DAYS,
                exit="tp1", trained_on=str(FOLDER.relative_to(ROOT)), n_trades=int(len(d)), n_h4=int(h4.sum()),
                h4_kept_pct=round(100 * float((sc[h4.to_numpy()] >= thr).mean()), 1),
                trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    (wf.MODELS / "h4_brain.json").write_text(json.dumps(meta, indent=1))
    print(f"h4_brain saved: {len(d):,} trades ({meta['n_h4']} H4), threshold {thr:.3f}, "
          f"H4 trades above it {meta['h4_kept_pct']}%")


if __name__ == "__main__":
    main()
