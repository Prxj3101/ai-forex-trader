r"""
CORRECTED WALK-FORWARD: same features, same ML settings, same bot filters,
but outcome labels from REAL TICKS (src/research/tick_labels.py). Writes NOTHING to models/ or ml_scores.

Rules fixed before running:
  - rows the ticks say never filled (no_fill) are dropped (~1%)
  - touch-1 rows where ticks show an EARLIER fill than the M5 data are dropped (~1%; real trade = different bar)
  - tick stop exits use the real tick price; +0.2 pip slippage and $5/lot are still added (same as before)
  - pass mark (decided in advance): bot net PF > 1.10 on unseen 2019-2026 and most years positive
Chalana:  python src\research\tick_walkforward.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wfm  # noqa: E402

LAB = ROOT / "data/research/tick_labels"
OUT = ROOT / "data/research/tick_v2"
KEY = ["symbol", "zone_tf", "zone_id", "touch_number", "fill_time"]


def build_dataset(sym):
    v = pd.read_parquet(ROOT / f"data/research/{sym}_v2.parquet")
    lab = pd.concat([pd.read_parquet(f) for f in sorted(LAB.glob(f"{sym}_*.parquet"))], ignore_index=True)
    d = v.merge(lab, on=KEY, how="left", validate="one_to_one")
    n0 = len(d)
    early = d["early_fill_ms"].notna() if "early_fill_ms" in d else False
    keep = (d["status"] == "ok") & ~early
    d = d[keep].copy()
    for p, col in (("tp1", "r_tp1"), ("tp2", "r_tp2"), ("trail", "r_trail")):
        d[col + "_m5"] = d[col]
        d[col] = d[f"t_{p}"]
        d[col + "_exit"] = d[f"t_{p}_why"]
    d["win"] = d["r_trail"] > 0
    d = d[[c for c in d.columns if not c.startswith("t_")]]
    OUT.mkdir(parents=True, exist_ok=True)
    d.drop(columns=[c for c in ("status", "early_fill_ms") if c in d]).to_parquet(OUT / f"{sym}_v2.parquet")
    return dict(rows=n0, kept=len(d), no_fill=int((lab.status == "no_fill").sum()), early=int(early.sum()))


if __name__ == "__main__":
    info = {s: build_dataset(s) for s in config.CORE_SYMBOLS}
    print(info, flush=True)
    which = sys.argv[1:] or ["tick"]
    out = {}
    for name in which:
        folder = OUT if name == "tick" else wfm.RESEARCH
        res, wf = wfm.train(config.CORE_SYMBOLS, folder=folder, save=False, log=lambda *a: None)
        out[name] = res
        b = res["bot"]
        print(name, "BOT:", {k: b[k] for k in ("trades", "win_pct", "pf", "total_r", "avg_r", "max_dd_r", "losing_years")})
        print("  by year:", b["by_year"]); print("  by pair:", b["by_symbol"])
        print("  summary:", [(r.get("keep"), r.get("trades"), r.get("pf_trail"), r.get("pf_1to1")) for r in res["summary"]])
        print("  score buckets:", [(x.get("bucket", x.get("group")), x.get("pf")) for x in res["score_buckets"]])
        wf.to_parquet(OUT / f"wf_{name}.parquet")
    (OUT / "tick_walkforward.json").write_text(json.dumps(out, indent=2, default=str))
