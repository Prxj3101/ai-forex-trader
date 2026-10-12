r"""
VISION BOT - build pictures + tick labels for ALL zones (run on the laptop: ticks live there).
One chunk per pair per year -> data/research/vision/build/{SYM}_{YEAR}.npz (images) + .parquet (labels/meta).
Already finished chunks are skipped, so it can be stopped (Ctrl+C) and started again.
Splits (fixed): train 2016-2020 | mock 2021-2022 | exam 2023-2026. purge = fill in the last 3 days before a split
boundary (a trade can run up to 3 days, so it would leak into the next period) -> never used.
The summary NEVER prints anything about the exam years (only counts).

Chalana:  python src\research\vision\vision_build.py build        (all chunks, 4 processes)
          python src\research\vision\vision_build.py summary      (checks + label mix train/mock)
          python src\research\vision\vision_build.py build 1 GBPUSD 2016 20   (smoke test: 1 process, 20 zones)
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_data as v  # noqa: E402

BUILD = v.OUT / "build"
YEARS = range(2016, 2027)
BOUNDS = [pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")]
COMM = 5.0  # $ per lot round trip


def split_of(t):
    y = t.year
    s = "train" if y <= 2020 else ("mock" if y <= 2022 else "exam")
    if any(b - pd.Timedelta(days=3) <= t < b for b in BOUNDS):
        s = "purge"
    return s


def chunk(job):
    sym, year, limit = job
    out = BUILD / f"{sym}_{year}"
    if out.with_suffix(".parquet").exists() and not limit:
        return f"{sym} {year}: already done"
    d = v.zones(years=[year])
    d = d[d.symbol == sym].sort_values("fill_ms").reset_index(drop=True)
    if limit:
        d = d.head(limit)
    imgs = np.zeros((len(d), 2, v.H, 4 * 3 * v.NB), np.uint8)
    rows, t0 = [], time.time()
    for i, r in d.iterrows():
        rec = dict(symbol=sym, direction=str(r.direction), zone_candle_time=r.zone_candle_time, created_at=r.created_at,
                   fill_time=r.fill_time, fill_ms=int(r.fill_ms), entry=r.entry, sl=r.sl, top=r.top, bottom=r.bottom,
                   r_tp1=r.r_tp1, r_tp2=r.r_tp2, r_trail=r.r_trail, r_tp1_exit=r.r_tp1_exit, r_tp2_exit=r.r_tp2_exit,
                   killzone=str(r.killzone), split=split_of(r.fill_time), err="")
        risk = abs(r.entry - r.sl)
        per_lot = risk * 1e5 if sym.endswith("USD") else risk * 1e5 / r.entry       # $ per 1R per lot
        rec["cost_r"] = COMM / per_lot
        try:
            lab = v.label(r)
            img, panels = v.picture(r)                        # asserts: no future bar / tick
            imgs[i] = img
            rec.update(label=lab["label"], best_r=lab["best_r"], why=lab["why"],
                       last_bar_close={tf: panels[tf].close_time.max() for tf in v.TFS}["M15"],
                       zoom_n=0 if panels["ZOOM"] is None else len(panels["ZOOM"]))
        except Exception as e:                                 # keep going, but record it
            rec.update(label=np.nan, err=f"{type(e).__name__}: {e}"[:200])
        rows.append(rec)
    meta = pd.DataFrame(rows)
    if limit:
        out = BUILD / f"_smoke_{sym}_{year}"
    BUILD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), img=imgs)
    meta.to_parquet(out.with_suffix(".parquet"))               # parquet written LAST = chunk finished
    return f"{sym} {year}: {len(d)} zones, errors {int((meta.err != '').sum())}, {time.time() - t0:.0f}s"


def build(procs=4, sym=None, year=None, limit=0):
    jobs = [(s, y, limit) for s in v.PAIRS for y in YEARS if (sym is None or s == sym) and (year is None or y == year)]
    if procs == 1:
        for j in jobs:
            print(chunk(j), flush=True)
        return
    from multiprocessing import Pool
    with Pool(procs) as p:
        for msg in p.imap_unordered(chunk, jobs):
            print(msg, flush=True)


def load_meta():
    fs = sorted(f for f in BUILD.glob("*.parquet") if not f.name.startswith("_"))
    return pd.concat([pd.read_parquet(f).assign(chunk=f.stem) for f in fs], ignore_index=True)


def summary():
    m = load_meta()
    print("chunks:", m.chunk.nunique(), "of", len(v.PAIRS) * len(YEARS), "| zones:", len(m))
    print("per split:", m.split.value_counts().to_dict())
    print("errors:", int((m.err != "").sum()), m[m.err != ""].err.str[:60].value_counts().head(5).to_dict())
    print("no zoom ticks:", int((m.zoom_n == 0).sum()))
    leak = (m.last_bar_close > m.fill_time).sum()
    print("LEAK CHECK last picture bar closed after fill:", int(leak), "(must be 0)")
    ok = m[m.label.notna()]
    agree = ((ok.label >= 1) == (ok.r_tp1_exit == "tp")) & ((ok.label >= 2) == (ok.r_tp2_exit == "tp"))
    print("LABEL CHECK vs v2 tick results:", int(agree.sum()), "/", len(ok), "agree")
    for s in ["train", "mock"]:                                # exam stays closed
        x = ok[ok.split == s]
        print(f"{s:5s} label mix %:", (x.label.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(), "| n", len(x))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build(procs=int(a[1]) if len(a) > 1 else 4, sym=a[2] if len(a) > 2 else None,
              year=int(a[3]) if len(a) > 3 else None, limit=int(a[4]) if len(a) > 4 else 0)
    elif a[0] == "summary":
        summary()
