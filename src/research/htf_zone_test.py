"""
HTF untapped BOS zone test (16 Oct 2026) - Prajwal: "purane zones sirf 1h / 4h ke, zone se nikla move pichle
(bade) swing ko tod ke BOS kare, zone untapped ho -> kitna bhi purana ho, valid".

RULES (run se PEHLE fix):
  - Zones: src/features/smc.find_orderflow_zones (BOS ki origin candle = bot ka hi zone), TF H1 aur H4,
    BOS reference swing n = 3 / 5 (bot abhi 1 use karta hai = chhota swing), displacement = config.disp_for(pair).
  - Untapped: zone created (BOS candle close) ke baad PEHLA touch hi trade. Life 60 trading din.
    Zone dead = M5 close zone ke wick ke paar (bot jaisa).
  - Entry: (a) WICK edge (bot ka config.ENTRY_LEVEL) aur (b) BODY edge. SL = zone ka door wala wick +- 0.1 ATR.
  - Exits M5 pe 1:1 / 1:2 / trail (management.run_trade), max 3 din. $5/lot + 0.2 pip.
  - Umar: 0-5 / 5-10 / 10-20 / 20-60 trading din. EURUSD / USDJPY / GBPUSD, 2016-20 aur 2021-26.
  - Pass: koi variant (ya "5+ din" wala hissa) NET PF > 1.05 DONO periods mein, aur 3 mein se 2 pairs > 1.

Chalana: python src/research/htf_zone_test.py run ; python src/research/htf_zone_test.py report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.features.smc import find_orderflow_zones  # noqa: E402
from src.research.management import POLICIES, run_trade  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402

OUT = ROOT / "data" / "research" / "htf_zone"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
LIFE, HOLD, BPD, BUF = 60, 3, 288, 0.1
POL = {p.name: p for p in POLICIES}
LABELS = {"r_tp1": POL["tp1"], "r_tp2": POL["tp2"], "r_trail": POL["trail1"]}
AGE_BINS, AGE_LAB = [0, 5, 10, 20, 60], ["0-5", "5-10", "10-20", "20-60"]


def zones(sym):
    out = []
    for tf in ("H1", "H4"):
        df = load(sym, tf)
        for n in (3, 5):
            z = find_orderflow_zones(df, swing_n=n, min_disp_atr=config.disp_for(sym))
            z["tf"], z["n"] = tf, n
            out.append(z)
    return pd.concat(out, ignore_index=True)


def run(sym):
    m5 = load(sym, "M5")
    lo, hi, cl, sp = (m5[c].to_numpy() for c in ["low", "high", "close", "spread"])
    idx = m5.index
    rows = []
    for r in zones(sym).itertuples(index=False):
        bull = r.direction == "bull"
        s = int(idx.searchsorted(r.created_at))
        e = min(s + LIFE * BPD, len(m5))
        if s >= e:
            continue
        stop = r.bottom - BUF * r.atr if bull else r.top + BUF * r.atr
        dead = (cl[s:e] < r.bottom) if bull else (cl[s:e] > r.top)
        dead_at = s + int(np.argmax(dead)) + 1 if dead.any() else e
        for entry_mode in ("wick", "body"):
            if bull:
                entry = r.top if entry_mode == "wick" else r.body_top
            else:
                entry = r.bottom if entry_mode == "wick" else r.body_bottom
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            fill = (lo[s:dead_at] + sp[s:dead_at] <= entry) if bull else (hi[s:dead_at] >= entry)
            hit = np.nonzero(fill)[0]
            if hit.size == 0:
                continue
            f = s + int(hit[0])
            g = min(f + HOLD * BPD, len(m5))
            row = dict(symbol=sym, tf=r.tf, n=r.n, entry_mode=entry_mode, direction=r.direction,
                       zone_candle_time=r.zone_candle_time, created_at=r.created_at, fill_time=idx[f],
                       age_days=(f - s) / BPD, entry=entry, sl=stop, risk=risk, risk_atr=risk / r.atr,
                       disp_atr=r.disp_atr, top=r.top, bottom=r.bottom)
            for col, p in LABELS.items():
                rr, why = run_trade(p, bull, entry, risk, hi[f:g], lo[f:g], cl[f:g], sp[f:g])
                row[col], row[col + "_exit"] = float(rr), why
            rows.append(row)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"{sym}.parquet")
    print(f"{sym}: {len(d):,} trades")


def pf(r):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def report():
    from src.ml.walkforward import add_costs, COMMISSION, SLIP_PIPS
    d = pd.concat([pd.read_parquet(OUT / f"{s}.parquet") for s in SYMS], ignore_index=True)
    d = add_costs(d, COMMISSION, SLIP_PIPS)
    d["period"] = np.where(d["fill_time"].dt.year <= 2020, "2016-20", "2021-26")
    d["age"] = pd.cut(d["age_days"], AGE_BINS, right=False, labels=AGE_LAB).astype(str)
    d["old"] = np.where(d["age_days"] >= 5, "5+", "0-5")
    rows = []
    keys = ["tf", "n", "entry_mode"]
    for k, g in d.groupby(keys):
        for part, gp in [("all", g)] + [(f"age {a}", x) for a, x in g.groupby("old")]:
            row = dict(zip(keys, k), part=part, trades=len(gp), per_year=round(len(gp) / 10.75),
                       win_1to1=round(100 * float((gp["r_tp1"] > 0).mean()), 1), gross_1to1=pf(gp["r_tp1_gross"]),
                       med_sl_pips=round(float(gp["sl_pips"].median()), 1))
            for per, x in gp.groupby("period"):
                for c, lab in (("r_tp1", "1to1"), ("r_tp2", "1to2"), ("r_trail", "trail")):
                    row[f"{lab}_{per[2:4]}-{per[-2:]}"] = pf(x[c])
            for s, x in gp.groupby("symbol"):
                row[f"1to1_{s[:3]}{s[3:]}"] = pf(x["r_tp1"])
            rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "report.csv", index=False)
    ages = d.groupby(["tf", "entry_mode", "age"])["r_tp1"].agg(["size", pf]).unstack("age")
    ages.to_csv(OUT / "report_age.csv")
    pd.set_option("display.width", 260)
    print(rep.to_string(index=False))
    print("\n1:1 by age (n 3+5 together):\n", ages.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    r = sp_.add_parser("run")
    r.add_argument("--symbol", nargs="+", default=SYMS)
    sp_.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        for s in a.symbol:
            run(s)
    else:
        report()
