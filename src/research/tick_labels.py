r"""
TICK LABELS: outcome labels for zone-dataset rows from REAL TICKS (src/backtest/tick_engine.py).

For each row (zone touch):
  1. real fill tick inside the row's fill candle [fill_time, fill_time + 5 min) (BUY ask <= entry, SELL bid >= entry).
     No tick -> "no_fill" (M5 data said fill, ticks say no) -> row removed.
  2. touch 1: was there an EARLIER real fill (between zone creation and this candle)? -> the real trade is a
     different candle -> row removed.
  3. exits from the fill tick, on ticks: r_tp1, r_tp2, r_trail (+ *_exit reasons). No ticks for that month
     -> the M5 label (with the fill-candle fix) stays and tick_status = "m5".
Used by zone_dataset_v2.build (tick_labels=True). For big rebuilds in small steps:
    python src\research\tick_labels.py EURUSD 2019      (labels of one year -> data/research/tick_labels/)
    python src\research\tick_labels.py merge EURUSD     (apply them to data/research/EURUSD_v2.parquet)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402

OUT = ROOT / "data" / "research" / "tick_labels"
KEY = ["symbol", "zone_tf", "zone_id", "touch_number", "fill_time"]
POL = {"r_tp1": "tp1", "r_tp2": "tp2", "r_trail": "trail"}


def ms(t: pd.Timestamp) -> int:
    return int(t.value // 1_000_000)


def label_rows(d: pd.DataFrame, sym: str) -> pd.DataFrame:
    rows = []
    for r in d.sort_values("fill_time").itertuples(index=False):
        bull = r.direction == "bull"
        f0 = ms(r.fill_time)
        out = dict(symbol=sym, zone_tf=r.zone_tf, zone_id=r.zone_id, touch_number=int(r.touch_number),
                   fill_time=r.fill_time)
        fill = te.find_fill(sym, bull, r.entry, f0, f0 + 300_000)
        if fill == "missing":
            rows.append(dict(out, status="missing")); continue
        if fill is None:
            rows.append(dict(out, status="no_fill")); continue
        out.update(status="ok", fill_ms=fill)
        if r.touch_number == 1:
            e = te.find_fill(sym, bull, r.entry, ms(r.created_at), f0)
            out["early_fill_ms"] = e if isinstance(e, int) else None
        for col, pol in POL.items():
            x = te.run_exit(sym, bull, r.entry, r.sl, fill, pol)
            out[f"t_{pol}"], out[f"t_{pol}_why"] = x["r"], x["reason"]
        rows.append(out)
    return pd.DataFrame(rows)


def apply(v: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Labels from ticks into the dataset. Removes no_fill / early-fill rows. Keeps M5 labels where no ticks."""
    d = v.drop(columns=[c for c in v.columns if c.startswith("t_") or c in ("status", "early_fill_ms", "fill_ms",
                                                                          "tick_status")], errors="ignore")
    d = d.merge(lab, on=KEY, how="left", validate="one_to_one")
    early = d["early_fill_ms"].notna() if "early_fill_ms" in d else pd.Series(False, index=d.index)
    drop = d["status"].isin(["no_fill"]) | early
    d = d[~drop].copy()
    ok = d["status"] == "ok"
    for col, pol in POL.items():
        d.loc[ok, col] = d.loc[ok, f"t_{pol}"].round(3)
        d.loc[ok, col + "_exit"] = d.loc[ok, f"t_{pol}_why"]
    d["tick_status"] = np.where(ok, "tick", "m5")
    d["win"] = d["r_trail"] > 0
    return d.drop(columns=[c for c in d.columns if c.startswith("t_") or c in ("status", "early_fill_ms")])


def relabel(v: pd.DataFrame, sym: str) -> pd.DataFrame:
    return apply(v, label_rows(v, sym))


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        sym = sys.argv[2]
        f = ROOT / f"data/research/{sym}_v2.parquet"
        v = pd.read_parquet(f)
        lab = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob(f"{sym}_*.parquet"))], ignore_index=True)
        d = apply(v, lab)
        d.to_parquet(f)
        print(sym, "rows", len(v), "->", len(d), d.tick_status.value_counts().to_dict())
    else:
        sym, year = sys.argv[1], int(sys.argv[2])
        v = pd.read_parquet(ROOT / f"data/research/{sym}_v2.parquet")
        r = label_rows(v[v.fill_time.dt.year == year], sym)
        OUT.mkdir(parents=True, exist_ok=True)
        r.to_parquet(OUT / f"{sym}_{year}.parquet")
        print(sym, year, len(r), r.status.value_counts().to_dict(), flush=True)
