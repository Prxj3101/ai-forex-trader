r"""
ORB BREAKEVEN TEST (9 Oct 2026, his question: "1R pe SL c2c karein toh?"). Rules fixed BEFORE running.
Trades = the 1,417 live-rule ORB trades 2019-26 (walk-forward ML + compression + 1 per pair per day,
data/research/orb_research.parquet status 'traded'), replayed on ticks from the break tick to 16:00 NY.
Exit variants (SL other side of the range at start; stop moves 1 s after the trigger tick, like the live bot):
  BASE      TP 2R (current)
  BE1       TP 2R, at +1R stop -> entry (0R)
  C2C1      TP 2R, at +1R stop -> entry + costs (~+0.1R, covers commission + spread)
  BE15      TP 2R, at +1.5R stop -> entry
  BE1_EOD   no TP, at +1R stop -> entry, close 16:00
  LOCK      TP 2R, at +1.5R stop -> +0.5R
Costs: $5/lot + 0.2 pip slippage on any stop exit (SL / BE / lock). Pass = net PF and total R better than BASE in BOTH halves.
Chalana: python src\research\orb_be_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
VARIANTS = {  # name: (tp_r or None, trigger_r or None, new stop_r)
    "BASE": (2.0, None, None), "BE1": (2.0, 1.0, 0.0), "C2C1": (2.0, 1.0, "cost"), "BE15": (2.0, 1.5, 0.0),
    "BE1_EOD": (None, 1.0, 0.0), "LOCK": (2.0, 1.5, 0.5)}


def sim(t, a, b, bull, entry, risk, tp, trig, new_stop):
    px = b if bull else a
    fav = ((px - entry) if bull else (entry - px)) / risk
    n = len(fav)
    stop = np.full(n, -1.0)
    if trig is not None:
        best = np.maximum.accumulate(fav)
        hit = np.nonzero(best >= trig)[0]
        if len(hit):
            act = np.searchsorted(t, t[hit[0]] + 1000)      # active 1 s later (bot updates once per second)
            stop[act:] = new_stop
    s_hit = np.nonzero(fav <= stop)[0]
    t_hit = np.nonzero(fav >= tp)[0] if tp is not None else np.array([], int)
    i_s = s_hit[0] if len(s_hit) else n
    i_t = t_hit[0] if len(t_hit) else n
    if i_s < n and i_s <= i_t:
        return float(fav[i_s]), ("sl" if stop[i_s] < -0.5 else "be")
    if i_t < n:
        return float(tp), "tp"
    return float(fav[-1]), "time"


def run():
    d = pd.read_parquet(ROOT / "data/research/orb_research.parquet")
    d = d[d.status == "traded"].sort_values("break_ms").reset_index(drop=True)
    rows = []
    for r in d.itertuples(index=False):
        bull = r.direction == "bull"
        tk = te.ticks(r.symbol, int(r.break_ms), int(r.eod) * 1000)
        if tk is None or not len(tk[0]):
            continue
        per_lot = r.risk * 1e5 if r.symbol.endswith("USD") else r.risk * 1e5 / r.entry
        comm_r = 5.0 / per_lot
        slip_r = 0.2 * PIP[r.symbol] / r.risk
        spread_r = float(tk[1][0] - tk[2][0]) / r.risk
        out = dict(symbol=r.symbol, range=r["range"] if isinstance(r, dict) else r.range, year=r.year, check=r.BRK_2R)
        for name, (tp, trig, ns) in VARIANTS.items():
            ns_ = (comm_r + slip_r + spread_r) if ns == "cost" else ns
            g, why = sim(*tk, bull, float(r.entry), float(r.risk), tp, trig, ns_)
            out[name] = g - comm_r - (slip_r if why in ("sl", "be") else 0)
            out[name + "_why"] = why
        rows.append(out)
    return pd.DataFrame(rows)


def pf(x):
    x = x.dropna()
    return round(float(x[x > 0].sum() / -x[x < 0].sum()), 3) if (x < 0).any() else np.nan


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    x = run()
    print("trades", len(x), "| BASE gross check vs backtest (should be ~0):", round(float((x.BASE + 0 - x.check).abs().median()), 3))
    rows = []
    for v in VARIANTS:
        c = x[v]
        rows.append(dict(variant=v, pf=pf(c), win=round(100 * float((c > 0).mean()), 1), total_r=round(float(c.sum()), 1),
                         avg_r=round(float(c.mean()), 3), pf_19_22=pf(x[x.year <= 2022][v]), pf_23_26=pf(x[x.year >= 2023][v]),
                         tp=int((x[v + "_why"] == "tp").sum()), be=int((x[v + "_why"] == "be").sum()),
                         sl=int((x[v + "_why"] == "sl").sum()), time=int((x[v + "_why"] == "time").sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    x.to_parquet(ROOT / "data/research/orb_be_test.parquet")
