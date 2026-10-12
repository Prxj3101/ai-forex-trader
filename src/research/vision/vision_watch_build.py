r"""
VISION BOT - Phase 2 "WATCH MODE" data: the bot does NOT enter at the zone touch. It WATCHES the reaction and
decides D minutes after the first touch (D = 5 or 15), entering at MARKET at the first tick >= decision time.
Rules (fixed before any result):
  touch     = the old limit-fill tick (first touch of the zone edge), from data/research/<SYM>_v2.parquet
  skip      if the SL is hit between touch and decision            -> status 'sl_during_watch' (no trade, like a human)
  entry     BUY at ask / SELL at bid of the first tick at/after decision time; SL = same zone SL
  geometry  risk = |entry - SL|; skip if risk < 0.5x or > 2x the zone risk (setup changed too much) -> 'geometry'
  picture   H4|H1|M15 bars CLOSED by decision time + M1 zoom from ticks strictly before the entry tick
            (so the zoom now SHOWS the reaction at the zone during the watch minutes)
  label     0-3 from ticks after entry (vision_data.label); net R for tp1/tp2/trail from tick_engine.run_exit
            (independent code path) - $5/lot commission, 0.2 pip slip on stop exits. Cross-check label vs run_exit.
Splits/purge exactly as Phase 1. Output: data/research/vision/build_w5 and build_w15 (same file layout).
Chalana:  python src\research\vision\vision_watch_build.py build 5        (then: build 15)
          python src\research\vision\vision_watch_build.py summary 5
          python src\research\vision\vision_watch_build.py smoke 5        (20 zones, 1 process)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_data as v  # noqa: E402
import vision_build as vb  # noqa: E402
from src.backtest.tick_engine import ticks, run_exit  # noqa: E402

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
POL = ["tp1", "tp2", "trail"]


def build_dir(delay):
    return v.OUT / f"build_w{delay}"


def watch_case(r, delay):
    """Returns (row for the watch trade or None, status)."""
    bull = str(r.direction) == "bull"
    touch = int(r.fill_ms)
    dec = touch + delay * 60_000
    z = ticks(r.symbol, touch, dec)
    if z is None:
        return None, "missing_ticks"
    t, a, b = z
    if len(t) and ((b <= r.sl).any() if bull else (a >= r.sl).any()):
        return None, "sl_during_watch"
    e = ticks(r.symbol, dec, dec + 30 * 60_000)
    if e is None or len(e[0]) == 0:
        return None, "no_entry_tick"
    te, ae, be = int(e[0][0]), float(e[1][0]), float(e[2][0])
    entry = ae if bull else be
    risk = entry - r.sl if bull else r.sl - entry
    orig = abs(r.entry - r.sl)
    if risk <= 0:
        return None, "beyond_sl"
    if not (0.5 <= risk / orig <= 2.0):
        return None, "geometry"
    w = r.copy()
    w["touch_ms"], w["orig_entry"], w["orig_risk"] = touch, r.entry, orig
    w["entry"], w["fill_ms"] = entry, te
    w["fill_time"] = pd.Timestamp(dec, unit="ms", tz="UTC")     # decision moment: bars must be closed by then
    return w, "ok"


def net_results(w):
    bull = str(w.direction) == "bull"
    risk = abs(w.entry - w.sl)
    per_lot = risk * 1e5 if w.symbol.endswith("USD") else risk * 1e5 / w.entry
    cost = vb.COMM / per_lot
    slip = 0.2 * PIP[w.symbol] / risk
    out = dict(cost_r=cost, risk=risk)
    for p in POL:
        res = run_exit(w.symbol, bull, w.entry, w.sl, int(w.fill_ms), p)
        out[f"g_{p}"], out[f"x_{p}"] = res["r"], res["reason"]
        out[f"n_{p}"] = res["r"] - cost - (slip if res["reason"] in ("sl", "trail", "be") else 0.0)
    return out


def chunk(job):
    sym, year, delay, limit = job
    bd = build_dir(delay)
    out = bd / (f"_smoke_{sym}_{year}" if limit else f"{sym}_{year}")
    if out.with_suffix(".parquet").exists() and not limit:
        return f"{sym} {year} w{delay}: already done"
    d = v.zones(years=[year])
    d = d[d.symbol == sym].sort_values("fill_ms").reset_index(drop=True)
    if limit:
        d = d.head(limit)
    imgs = np.zeros((len(d), 2, v.H, 4 * 3 * v.NB), np.uint8)
    rows, t0 = [], time.time()
    for i, r in d.iterrows():
        rec = dict(symbol=sym, direction=str(r.direction), zone_candle_time=r.zone_candle_time, touch_ms=int(r.fill_ms),
                   touch_time=r.fill_time, sl=r.sl, top=r.top, bottom=r.bottom, killzone=str(r.killzone),
                   delay=delay, split=vb.split_of(r.fill_time), err="", label=np.nan)
        try:
            w, status = watch_case(r, delay)
            rec["status"] = status
            if w is not None:
                lab = v.label(w)
                img, panels = v.picture(w)                     # asserts: no bar closed after decision, no tick at/after entry
                imgs[i] = img
                rec.update(entry=w.entry, orig_entry=w.orig_entry, orig_risk=w.orig_risk, fill_ms=int(w.fill_ms),
                           fill_time=w.fill_time, label=lab["label"], best_r=lab["best_r"], why=lab["why"],
                           last_bar_close=panels["M15"].close_time.max(),
                           zoom_n=0 if panels["ZOOM"] is None else len(panels["ZOOM"]), **net_results(w))
        except Exception as e:
            rec.update(status="error", err=f"{type(e).__name__}: {e}"[:200])
        rows.append(rec)
    meta = pd.DataFrame(rows)
    bd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), img=imgs)
    meta.to_parquet(out.with_suffix(".parquet"))
    st = meta.status.value_counts().to_dict()
    return f"{sym} {year} w{delay}: {len(d)} touches -> {st}, {time.time() - t0:.0f}s"


def build(delay, procs=4, limit=0, sym=None, year=None):
    jobs = [(s, y, delay, limit) for s in v.PAIRS for y in vb.YEARS if (sym is None or s == sym) and (year is None or y == year)]
    if procs == 1:
        for j in jobs:
            print(chunk(j), flush=True)
        return
    from multiprocessing import Pool
    with Pool(procs) as p:
        for msg in p.imap_unordered(chunk, jobs):
            print(msg, flush=True)


def load_meta(delay, smoke=False):
    fs = sorted(f for f in build_dir(delay).glob("*.parquet") if f.name.startswith("_smoke") == smoke)
    return pd.concat([pd.read_parquet(f).assign(chunk=f.stem) for f in fs], ignore_index=True)


def summary(delay, smoke=False):
    m = load_meta(delay, smoke)
    print(f"WATCH {delay} min | chunks {m.chunk.nunique()} | touches {len(m)}")
    print("status:", m.status.value_counts().to_dict())
    print("errors:", m[m.status == "error"].err.str[:70].value_counts().head(5).to_dict())
    ok = m[m.status == "ok"].copy()
    print("per split (ok):", ok.split.value_counts().to_dict())
    print("LEAK CHECK bar closed after decision:", int((ok.last_bar_close > ok.fill_time).sum()), "(must be 0)")
    print("ENTRY AFTER DECISION:", int((ok.fill_ms < ok.touch_ms + delay * 60_000).sum()), "violations (must be 0)")
    agree = ((ok.label >= 1) == (ok.x_tp1 == "tp")) & ((ok.label >= 2) == (ok.x_tp2 == "tp"))
    print("LABEL CHECK label vs tick_engine.run_exit:", int(agree.sum()), "/", len(ok))
    print("risk / zone risk  p5 p50 p95:", ok.eval("risk / orig_risk").quantile([.05, .5, .95]).round(2).tolist())
    for s in ["train", "mock"]:                                # exam stays closed
        x = ok[ok.split == s]
        pfs = {p: round(float(x[f"n_{p}"][x[f"n_{p}"] > 0].sum() / -x[f"n_{p}"][x[f"n_{p}"] < 0].sum()), 3) for p in POL}
        print(f"{s:5s} n {len(x)} | label mix %", (x.label.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
              "| take-all net PF", pfs)


if __name__ == "__main__":
    a = sys.argv[1:]
    dly = int(a[1])
    if a[0] == "build":
        build(dly, procs=int(a[2]) if len(a) > 2 else 4)
    elif a[0] == "smoke":
        build(dly, procs=1, limit=20, sym="GBPUSD", year=2016); summary(dly, smoke=True)
    elif a[0] == "summary":
        summary(dly)
