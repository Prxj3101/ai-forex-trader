r"""
LIVE BRAIN: saved ML models se live trade cards ko score karna.

  models/zone_brain.txt  -> "is trade ke jeetne ka chance" (score 0-1)
  models/exit_brain.txt  -> "trailing 1:1 se kitna behtar hoga" (> 0 = trail, warna 1:1)
  data/research/ml_walkforward.json -> final_thresholds["0.3"] = top 30% ka cut-off

Ye files dashboard ke RETRAIN (ya python src\ml\walkforward.py) se banti hain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml import walkforward as wf  # noqa: E402

MODELS = ROOT / "models"
REPORT = ROOT / "data" / "research" / "ml_walkforward.json"
KEEP = "0.3"            # bot sirf ML ke top 30% trades leta hai


class Brain:
    def __init__(self):
        for f in (MODELS / "zone_brain.txt", MODELS / "exit_brain.txt", REPORT):
            if not f.exists():
                raise SystemExit(f"{f.name} nahi mila. Pehle ML train karo (dashboard RETRAIN ya python src\\ml\\walkforward.py).")
        rep = json.loads(REPORT.read_text())
        thr = rep.get("final_thresholds", {}).get(KEEP)
        if thr is None:
            raise SystemExit("ml_walkforward.json mein final_thresholds nahi hai (purana training). Ek baar RETRAIN karo.")
        self.threshold = float(thr)
        self.trained_at = rep.get("trained_at")
        self.entry = lgb.Booster(model_file=str(MODELS / "zone_brain.txt"))
        self.exit = lgb.Booster(model_file=str(MODELS / "exit_brain.txt"))
        self.features = self.entry.feature_name()
        missing = set(self.features) - set(wf.FEATURES)
        if missing:
            raise SystemExit(f"Model mein features hain jo code mein nahi: {missing}. RETRAIN karo.")

    def score(self, d: pd.DataFrame) -> pd.DataFrame:
        """d = live_features.candidates() output. Adds score, ml_keep, exit_choice, exit_edge."""
        if d.empty:
            return d
        X = d[self.features]
        d = d.copy()
        d["score"] = self.entry.predict(X)
        d["ml_keep"] = d["score"] >= self.threshold
        edge = self.exit.predict(X)
        d["exit_edge"] = edge
        d["exit_choice"] = np.where(edge > 0, "trail", "tp1")
        return d


class H4Brain:
    """16 Oct 2026: H4 zone trades only (models/h4_brain.txt + h4_brain.json, src/ml/h4_brain.py).
    Exit is always 1:1 (research: exit brain hurt the H4 trades). Missing files -> H4 trades OFF (bot keeps running)."""

    def __init__(self):
        f, meta = MODELS / "h4_brain.txt", MODELS / "h4_brain.json"
        if not f.exists() or not meta.exists():
            raise FileNotFoundError("models/h4_brain.txt / h4_brain.json missing (python src/ml/h4_brain.py)")
        m = json.loads(meta.read_text())
        self.threshold = float(m["threshold"])
        self.trained_at = m.get("trained_at")
        self.entry = lgb.Booster(model_file=str(f))
        self.features = self.entry.feature_name()

    def score(self, d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        d = d.copy()
        d["score"] = self.entry.predict(d[self.features])
        d["ml_keep"] = d["score"] >= self.threshold
        d["exit_edge"] = 0.0
        d["exit_choice"] = "tp1"
        return d
