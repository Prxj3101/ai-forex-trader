r"""
ORB REPLAY CHECK: live ORB engine (src/live/orb.py) on old data (ReplayFeed: Dukascopy M5 + ticks) vs the backtest
(data/research/orb_ml_features.parquet). Same days must give the same ranges, compression, breaks, entries, features,
ML score (final model) and trade R.
Chalana: python src\live\orb_replay_check.py 2026-03-02 2026-03-14
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.live.feeds import ReplayFeed  # noqa: E402
from src.live.orb import OrbEngine, OrbBrain  # noqa: E402
from src.research import orb_ml_test as om  # noqa: E402

OUT = Path.home() / "orb_replay"


def run(start, end):
    if OUT.exists():
        shutil.rmtree(OUT)
    feed = ReplayFeed(config.CORE_SYMBOLS, start, end, 200)
    eng = OrbEngine(feed, out_dir=OUT, resume=False, write_view=False, log=lambda *a: None)
    eng.keep_days = 10_000
    t0 = time.time()
    for _ in feed:
        eng.step()
    print(f"replay {start}->{end}: {time.time() - t0:.0f}s, closed {eng.n_closed}, open {len(eng.open)}")
    return eng


def compare(eng, start, end):
    recs = pd.DataFrame(list(eng.recs.values()))
    bt = pd.read_parquet(om.FEAT)
    bt["day"] = bt["day"].astype(str)
    bt = bt[(bt.day >= start) & (bt.day < end)]
    m = recs.merge(bt, on=["day", "symbol", "range"], how="outer", suffixes=("", "_bt"), indicator=True)
    print("ranges live:", len(recs), "| backtest breaks:", len(bt), "| merge:", m._merge.value_counts().to_dict())
    both = m[m._merge == "both"]
    brk = both[both.break_ms.notna()]
    print("break time match:", int((brk.break_ms == brk.break_ms_bt).sum()), "/", len(brk),
          "| direction match:", int((brk.direction == brk.direction_bt).sum()))
    # features at break (live) vs backtest
    br = OrbBrain()
    feats = [f for f in om.FEATURES if f not in ("symbol", "range")]
    rows = []
    for _, r in brk.iterrows():
        f = r["features"] if isinstance(r.get("features"), dict) else None
        if f is None:                    # skipped before break (no compression): no live features
            continue
        diff = {k: (f.get(k), r[k]) for k in feats if not (pd.isna(f.get(k)) and pd.isna(r[k]))
                and (pd.isna(f.get(k)) or pd.isna(r[k]) or abs(float(f[k]) - float(r[k])) > 1e-6 * max(1, abs(float(r[k]))))}
        sbt = br.score({**{k: r[k] for k in feats}, "symbol": r.symbol, "range": r["range"]})
        rows.append(dict(day=r.day, symbol=r.symbol, range=r["range"], live_score=r.score, bt_score=round(sbt, 4),
                         n_diff=len(diff), diff=diff))
    t = pd.DataFrame(rows)
    if len(t):
        print("feature rows compared:", len(t), "| rows with any feature diff:", int((t.n_diff > 0).sum()),
              "| max score diff:", round(float((t.live_score - t.bt_score).abs().max()), 4))
        bad = t[t.n_diff > 0]
        if len(bad):
            from collections import Counter
            print("features that differ:", Counter(k for d in bad["diff"] for k in d))
            print(bad.head(8).to_string())
    # expected trades from the backtest rows with the FINAL model: compression + score >= threshold, earliest per pair/day
    from src.research.orb_train import tight_cutoffs
    q = tight_cutoffs()
    bt = bt.copy()
    bt["score_final"] = [br.score({**{k: r[k] for k in feats}, "symbol": r.symbol, "range": r["range"]}) for _, r in bt.iterrows()]
    comp = (bt.nr4 == 1) | (bt.w_ratio <= bt["range"].map(q))
    exp = bt[comp & (bt.score_final >= br.threshold)].sort_values("break_ms").groupby(["symbol", "day"]).head(1)
    nr_rows = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/research/orb_nr").glob("nr_*.parquet"))])
    nr_rows["day"] = nr_rows["day"].astype(str)
    exp = exp.merge(nr_rows[["symbol", "day", "range", "BRK_2R"]], on=["symbol", "day", "range"], how="left")
    # trades
    tr = pd.read_csv(OUT / "orb_trades.csv") if (OUT / "orb_trades.csv").exists() else pd.DataFrame()
    if len(tr):
        tr["day"] = tr["day"].astype(str)
        k = ["day", "symbol", "range"]
        x = tr[k + ["direction", "entry", "r", "exit_reason"]].merge(exp[k + ["BRK_2R"]], on=k, how="outer", indicator=True)
        x["r_diff"] = (x.r - x.BRK_2R).abs()
        print("\nlive trades:", len(tr), "| expected (backtest + final model):", len(exp), "|", x._merge.value_counts().to_dict(),
              "| max |R live - R backtest| (gross):", round(float(x.r_diff.max()), 4))
        print(x[x._merge != "both"].to_string(index=False) if (x._merge != "both").any() else "all trades match")
        print(x.sort_values("r_diff", ascending=False).head(5).to_string(index=False))
    return t


if __name__ == "__main__":
    s, e = sys.argv[1], sys.argv[2]
    compare(run(s, e), s, e)
