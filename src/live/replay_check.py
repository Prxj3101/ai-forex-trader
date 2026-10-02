r"""
REPLAY CHECK: kya LIVE code wahi dekhta hai jo BACKTEST ne dekha tha?

Tareeka: backtest dataset (data/research/{SYM}_v2.parquet) se M15 pehle-touch trades lo.
Har trade ke fill time T pe live engine ko SIRF T se pehle ka data do (pichle 120 din, jaise live mein
MT5 se milega) aur poocho: "abhi kaunse zones trade ke layak hain, unke features kya hain?"
Phir dono ko milao: same zone mila? Har feature same hai?

Mismatch = live bot backtest se alag behave karega -> demo se pehle theek karna zaroori.

Run: python src\live\replay_check.py --symbol EURUSD --n 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.live import live_features as lf  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402

DATA = ROOT / "data" / "processed"
RESEARCH = ROOT / "data" / "research"
CHECK = [c for c in wf.FEATURES if c not in ("symbol", "zone_tf")] + ["entry", "sl", "r24_va_pos", "act_15m"]


def run(symbol: str, n: int = 40, since: str = "2026-01-01", seed: int = 0, log=print) -> pd.DataFrame:
    m5 = pd.read_parquet(DATA / f"{symbol}_M5.parquet")
    ds = wf.load([symbol])                                  # backtest rows with derived features (same as training)
    ds = ds[(ds["zone_tf"].astype(str) == "M15") & (ds["touch_number"] == 1) & (ds["fill_time"] >= since)]
    ds = ds.sample(min(n, len(ds)), random_state=seed).sort_values("fill_time")
    out = []
    t0 = time.time()
    for r in ds.itertuples(index=False):
        T = r.fill_time
        w = m5[(m5.index < T) & (m5.index >= T - pd.Timedelta(days=lf.HISTORY_DAYS))]
        c = lf.candidates(symbol, w, T)
        hit = c[(c["zone_candle_time"] == r.zone_candle_time) & (c["direction"].astype(str) == str(r.direction))] if len(c) else c
        row = dict(fill_time=T, zone=r.zone_candle_time, direction=r.direction, found=len(hit) == 1, live_candidates=len(c))
        if len(hit) == 1:
            h = hit.iloc[0]
            diffs = {}
            for col in CHECK:
                a, b = getattr(r, col), h[col]
                if isinstance(a, (int, float, np.floating, np.integer, bool, np.bool_)) and not isinstance(a, str):
                    a, b = float(a), float(b)
                    same = (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-4 * max(1, abs(a))   # 0.01%: ATR warm-up se 5th decimal ka farak OK
                else:
                    same = str(a) == str(b)
                if not same:
                    diffs[col] = (a, b)
            row["n_diff"] = len(diffs)
            row["diffs"] = "; ".join(f"{k}: bt={v[0]:.5g} live={v[1]:.5g}" if isinstance(v[0], float) else f"{k}: bt={v[0]} live={v[1]}"
                                     for k, v in diffs.items())
        out.append(row)
    res = pd.DataFrame(out)
    log(f"{symbol}: {len(res)} trades checked in {time.time() - t0:.0f}s | zone found {res['found'].mean():.0%} | "
        f"all features equal {(res.get('n_diff', pd.Series(dtype=float)) == 0).mean():.0%}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--since", default="2026-01-01")
    a = ap.parse_args()
    r = run(a.symbol, a.n, a.since)
    pd.set_option("display.width", 220, "display.max_colwidth", 200)
    bad = r[(~r["found"]) | (r.get("n_diff", 0) > 0)]
    print(bad.to_string() if len(bad) else "ALL MATCH")
