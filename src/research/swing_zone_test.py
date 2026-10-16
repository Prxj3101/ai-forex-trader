"""
Swing supply / demand zone test (16 Oct 2026) - Prajwal ka apna zone (swing high = supply, swing low = demand),
purane untapped zones ke saath. Bot ka zone (BOS origin candle) se alag cheez.

RULES (run se PEHLE fix, baad mein nahi badlenge):
  - Zone TF: M15 aur H1. Swing = fractal, n = 3 aur n = 5 (dono taraf n candles se ooncha / neecha).
    Swing tabhi pata jab right side ki n candles band ho jaayein (no lookahead).
  - Supply (swing high candle): zone = [max(open, close), high]. SELL LIMIT = max(open, close) (body edge),
    SL = high + 0.1 ATR. Demand mirror: BUY LIMIT = min(open, close), SL = low - 0.1 ATR.
  - Clean move (uska "clean move"): zone active tabhi jab price pehle entry se K ATR door chala jaye
    (K = 2 aur 4, HTF ATR14). Agar K ATR door jaane se PEHLE hi entry wapas chhu li -> zone discard.
  - Sirf PEHLA touch. Life: 40 trading din tak (= practically "jab tak tap ya break na ho");
    result zone ki umar ke hisaab se: 0-5 / 5-10 / 10-20 / 20-40 din.
  - Risk filter: SL distance 0.25 - 3 ATR (bahut chhota SL = sirf kharcha).
  - Fill + exits M5 pe (src/research/management.run_trade, fill candle fix), max hold 3 din.
    Exits: 1:1 (uska normal), 1:2, trail. Costs: $5/lot + 0.2 pip slippage (walkforward.add_costs).
  - Pairs: EURUSD, USDJPY, GBPUSD. Periods: 2016-20 aur 2021-26 alag.
  - Verdict: koi variant tabhi "kaam ka" jab 1:1 NET PF > 1.05 DONO periods mein, aur pados ke variants bhi > 1.

Chalana:  python src/research/swing_zone_test.py run --symbol EURUSD   (har pair alag, ~1 min)
          python src/research/swing_zone_test.py report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.smc import atr, find_swings  # noqa: E402
from src.research.management import POLICIES, run_trade  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402

OUT = ROOT / "data" / "research" / "swing_zone"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
TFS = ["M15", "H1"]
NS = [3, 5]
KS = [2.0, 4.0]
LIFE_DAYS, HOLD_DAYS, BPD = 40, 3, 288
BUF = 0.1
POL = {p.name: p for p in POLICIES}
LABELS = {"r_tp1": POL["tp1"], "r_tp2": POL["tp2"], "r_trail": POL["trail1"]}
AGE_BINS = [0, 5, 10, 20, 40]
AGE_LAB = ["0-5", "5-10", "10-20", "20-40"]


def run(sym: str) -> Path:
    m5 = load(sym, "M5")
    lo, hi, cl, sp = (m5[c].to_numpy() for c in ["low", "high", "close", "spread"])
    idx = m5.index
    rows = []
    for tf in TFS:
        df = load(sym, tf)
        a = atr(df).to_numpy()
        o, c, h, l = (df[x].to_numpy() for x in ["open", "close", "high", "low"])
        for n in NS:
            sh, sl_ = find_swings(df, n)
            for j in np.nonzero(sh | sl_)[0]:
                if j + n + 1 >= len(df) or not np.isfinite(a[j]):
                    continue
                for bull in ([False] if sh[j] and not sl_[j] else [True] if sl_[j] and not sh[j] else [False, True]):
                    if bull:
                        entry, stop = min(o[j], c[j]), l[j] - BUF * a[j]
                    else:
                        entry, stop = max(o[j], c[j]), h[j] + BUF * a[j]
                    risk = abs(entry - stop)
                    if not (0.25 * a[j] <= risk <= 3 * a[j]):
                        continue
                    s = int(idx.searchsorted(df.index[j + n + 1]))      # swing confirm hone ke baad
                    e = min(s + LIFE_DAYS * BPD, len(m5))
                    if s >= e:
                        continue
                    touch = (lo[s:e] + sp[s:e] <= entry) if bull else (hi[s:e] >= entry)
                    t_any = np.nonzero(touch)[0]
                    for K in KS:
                        away = (hi[s:e] >= entry + K * a[j]) if bull else (lo[s:e] <= entry - K * a[j])
                        dep = np.nonzero(away)[0]
                        if dep.size == 0:
                            continue
                        d0 = int(dep[0])
                        if t_any.size and t_any[0] <= d0:                # chhu liya clean move se pehle
                            continue
                        after = t_any[t_any > d0]
                        if after.size == 0:
                            continue
                        f = s + int(after[0])
                        g = min(f + HOLD_DAYS * BPD, len(m5))
                        row = dict(symbol=sym, tf=tf, n=n, K=K, direction="bull" if bull else "bear",
                                   swing_time=df.index[j], active_time=idx[s + d0], fill_time=idx[f],
                                   age_days=(f - s) / BPD, entry=entry, sl=stop, risk=risk, risk_atr=risk / a[j])
                        for col, p in LABELS.items():
                            r, why = run_trade(p, bull, entry, risk, hi[f:g], lo[f:g], cl[f:g], sp[f:g])
                            row[col], row[col + "_exit"] = float(r), why
                        rows.append(row)
        print(f"{sym} {tf}: {len(rows):,} trades so far")
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{sym}.parquet"
    d.to_parquet(f)
    print(f"saved {f.relative_to(ROOT)} ({len(d):,} trades)")
    return f


def pf(r):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def report() -> None:
    from src.ml.walkforward import add_costs, COMMISSION, SLIP_PIPS
    d = pd.concat([pd.read_parquet(OUT / f"{s}.parquet") for s in SYMS], ignore_index=True)
    d = add_costs(d, COMMISSION, SLIP_PIPS)
    d["period"] = np.where(d["fill_time"].dt.year <= 2020, "2016-20", "2021-26")
    d["age"] = pd.cut(d["age_days"], AGE_BINS, right=False, labels=AGE_LAB)
    rows = []
    keys = ["tf", "n", "K"]
    for k, g in d.groupby(keys):
        base = dict(zip(keys, k))
        for per, gp in [("all", g)] + list(g.groupby("period")):
            rows.append(dict(**base, period=per, age="all", trades=len(gp),
                             per_year=round(len(gp) / (10.75 if per == "all" else 5 if per == "2016-20" else 5.75), 0),
                             pf_1to1=pf(gp["r_tp1"]), pf_1to2=pf(gp["r_tp2"]), pf_trail=pf(gp["r_trail"]),
                             avg_r_1to1=round(float(gp["r_tp1"].mean()), 3)))
            for age, ga in gp.groupby("age", observed=True):
                rows.append(dict(**base, period=per, age=str(age), trades=len(ga), per_year=np.nan,
                                 pf_1to1=pf(ga["r_tp1"]), pf_1to2=pf(ga["r_tp2"]), pf_trail=pf(ga["r_trail"]),
                                 avg_r_1to1=round(float(ga["r_tp1"].mean()), 3)))
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "report.csv", index=False)
    sym = d.groupby(["tf", "n", "K", "symbol", "period"])["r_tp1"].apply(pf).unstack(["symbol", "period"])
    sym.to_csv(OUT / "report_by_symbol.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(rep.to_string(index=False))
    print("\n1:1 PF by pair:\n", sym.to_string())


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
