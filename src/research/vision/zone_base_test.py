r"""
ZONE BASE-TRADE TEST (11 Oct) - can the TRADE itself get near PF 1.0? (no ML, no mock tuning)
Base = watch-5 trades (status ok) with min SL 8 pips (picked on fit 2016-19 in vision_v3), 2016-2022 only (exam untouched).
Variations (fixed BEFORE results, max 4 + base; all net: $5/lot commission on the NEW risk + 0.2 pip slip on stop exits):
  V0 base       SL = zone SL, TP 2R
  V1 buf025     SL = zone SL pushed 0.25 x ATR(zone) further away, TP 2R (of the new, bigger risk)
  V2 buf050     SL pushed 0.50 x ATR, TP 2R
  V3 tp15       SL = zone SL, TP 1.5R
  V4 killzone   V0 but only touches in London open / NY open
PASS (fixed now): a variation with PF >= 0.95 in at least 3 of the 4 years 2019-2022 AND pooled 2019-22 PF >= 1.00.
  -> then cnn_pre is retrained on that trade once, then the exam. Otherwise zones closed for good.
Chalana:  python src\research\vision\zone_base_test.py run        (4 processes, ~10-20 min)
          python src\research\vision\zone_base_test.py report
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import vision_data as vd  # noqa: E402
from src.backtest.tick_engine import run_exit  # noqa: E402

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
OUT = vd.OUT / "base_test"
MIN_SL = 8
YEARS = range(2016, 2023)
VARS = {"V0_base": (0.0, 2.0, False), "V1_buf025": (0.25, 2.0, False), "V2_buf050": (0.5, 2.0, False),
        "V3_tp15": (0.0, 1.5, False), "V4_killzone": (0.0, 2.0, True)}


def trades(sym, year):
    m = pd.read_parquet(vd.OUT / "build_w5" / f"{sym}_{year}.parquet")
    m = m[(m.status == "ok") & m.split.isin(["train", "mock"])].copy()
    m = m[m.risk / PIP[sym] >= MIN_SL]
    v = pd.read_parquet(ROOT / f"data/research/{sym}_v2.parquet", columns=["fill_ms", "atr", "zone_tf", "touch_number"])
    v = v[(v.zone_tf.astype(str) == "M15") & (v.touch_number == 1)].drop_duplicates("fill_ms", keep="last")
    m = m.merge(v[["fill_ms", "atr"]].rename(columns={"fill_ms": "touch_ms"}), on="touch_ms", how="left")
    assert m.atr.notna().all(), "atr merge failed"
    return m


def one(job):
    sym, year = job
    f = OUT / f"{sym}_{year}.parquet"
    if f.exists():
        return f"{sym} {year}: already done"
    m, t0, rows = trades(sym, year), time.time(), []
    for _, r in m.iterrows():
        bull = r.direction == "bull"
        rec = dict(symbol=sym, year=year, fill_time=r.fill_time, killzone=r.killzone)
        for name, (buf, tp, kz) in VARS.items():
            sl = r.sl - buf * r.atr if bull else r.sl + buf * r.atr
            risk = abs(r.entry - sl)
            per_lot = risk * 1e5 if sym.endswith("USD") else risk * 1e5 / r.entry
            res = run_exit(sym, bull, r.entry, sl, int(r.fill_ms), "tp", tp)
            net = res["r"] - 5.0 / per_lot - (0.2 * PIP[sym] / risk if res["reason"] in ("sl", "trail", "be") else 0.0)
            if kz and r.killzone not in ("London open", "NY open"):
                net = np.nan                                       # filtered out = no trade
            rec[name] = net
        rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f)
    return f"{sym} {year}: {len(m)} trades, {time.time() - t0:.0f}s"


def run(procs=4):
    jobs = [(s, y) for s in vd.PAIRS for y in YEARS]
    from multiprocessing import Pool
    with Pool(procs) as p:
        for msg in p.imap_unordered(one, jobs):
            print(msg, flush=True)


def pf(x):
    x = x.dropna()
    return x[x > 0].sum() / -x[x < 0].sum() if (x < 0).any() else np.nan


def report():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("*.parquet"))], ignore_index=True)
    # sanity: V0 must equal the tp2 result already in build_w5 (same entry, SL, TP 2R, costs)
    t = pd.DataFrame({v: d.groupby("year")[v].apply(pf) for v in VARS}).round(3)
    t["trades_V0"] = d.groupby("year").V0_base.count()
    print("NET PF per year (2016-18 = context, 2019-22 = the test):\n" + t.to_string())
    wf = d[d.year.between(2019, 2022)]
    print("\nPASS CHECK (>= 3 of 4 years PF >= 0.95 AND pooled 2019-22 PF >= 1.00):")
    for v in VARS:
        yrs = [pf(wf[wf.year == y][v]) for y in range(2019, 2023)]
        good = sum(p >= 0.95 for p in yrs); pooled = pf(wf[v])
        print(f"  {v:12s} years >=0.95: {good}/4 | pooled PF {pooled:.3f} | trades {int(wf[v].count())} | total R {wf[v].sum():.0f} -> "
              f"{'PASS' if good >= 3 and pooled >= 1.0 else 'FAIL'}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
