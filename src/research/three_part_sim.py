"""
3-part zone bot account replay (16 Oct 2026). Prajwal ka plan:
  15m zone bot (bina confluence) 0.25% | 15m + H4 zone confluence 1% | H4 zone trades 0.75%.
Trades = out-of-sample walk-forward (2019-26): confluence/wf_base_h4flag.parquet (M15, exit brain) +
zone_h4/wf_n5.parquet (H4 touch1, top30 + VA + act, 1:1). Account rules = portfolio_sim.simulate
(max 3 open, 1 per pair, max 2 same USD side, daily loss 2%, daily risk budget 2.5%, brake 5% DD -> half,
kill 10% DD -> 7 days pause, $5/lot + 0.2 pip slippage). Fixed risk (no house money), max 1% per trade.
Note: M5 labels (no tick replay). Run: python src/research/three_part_sim.py
"""
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.risk import portfolio_sim as ps  # noqa: E402
from src.research.management import POLICIES, run_trade_full  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402

POL = {p.name: p for p in POLICIES}
PLANS = {"now (all 15m 0.5%)": dict(plain=0.5, conf=0.5, h4=0.0),
         "his plan 0.25 / 1 / 0.75": dict(plain=0.25, conf=1.0, h4=0.75)}

# simulate() with a per-trade risk multiplier (c.mult)
src = inspect.getsource(ps.simulate).replace(
    "risk_pct, braked = trade_risk_pct(cfg, equity, peak, month_start)",
    "risk_pct, braked = trade_risk_pct(cfg, equity, peak, month_start)\n        risk_pct = risk_pct * c.mult")
src = src.replace("r=c.r, pnl=pnl,", "r=c.r, pnl=pnl, kind=c.kind,")
ns = dict(vars(ps))
exec(src, ns)
simulate = ns["simulate"]


def cands():
    w = pd.read_parquet(ROOT / "data/research/confluence/wf_base_h4flag.parquet")
    b = w[(w.zone_tf == "M15") & (w.touch_number == 1) & w["keep_0.3"] & w.va_ok & w.act_ok].copy()
    b["kind"] = np.where(b.h4_zone, "conf", "plain")
    b["r"], b["exit"] = b["r_ml"], b["exit_ml"]
    h = pd.read_parquet(ROOT / "data/research/zone_h4/wf_n5.parquet")
    h = h[(h.zone_tf == "H4") & (h.touch_number == 1) & h["keep_0.3"] & h.va_ok & h.act_ok].copy()
    h["kind"], h["r"], h["exit"] = "h4", h["r_tp1"], "tp1"
    x = pd.concat([b, h], ignore_index=True)
    x["symbol"], x["direction"] = x["symbol"].astype(str), x["direction"].astype(str)
    rows = []
    for sym, g in x.groupby("symbol"):
        m5 = load(sym, "M5")
        hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
        idx = m5.index
        for r in g.itertuples(index=False):
            f = int(idx.searchsorted(r.fill_time))
            gg = min(f + 3 * 288, len(m5))
            bull = r.direction == "bull"
            pol = POL["tp1"] if r.exit == "tp1" else POL["trail1"]
            rr, why, k = run_trade_full(pol, bull, r.entry, r.risk, hi[f:gg], lo[f:gg], cl[f:gg], sp[f:gg])
            # r from the dataset already has costs; recompute gross from the same replay (simulate adds costs)
            rows.append(dict(symbol=sym, direction=r.direction, fill_time=r.fill_time,
                             exit_time=idx[f + k] + pd.Timedelta(minutes=5), entry=r.entry, risk=r.risk,
                             exit=r.exit, r=rr, reason=why, kind=r.kind))
    return pd.DataFrame(rows).sort_values("fill_time").reset_index(drop=True)


def main():
    c = cands()
    cfg = replace(ps.RiskConfig(), capital=10000, risk_pct=1.0, risk_cap_pct=1.0, house_money_pct=0.0,
                  daily_loss_pct=2.0, daily_risk_budget_pct=2.5, kill_dd_pct=10.0, commission_per_lot=5.0)
    out = []
    for name, m in PLANS.items():
        cc = c.copy()
        cc["mult"] = cc["kind"].map(m)
        cc = cc[cc["mult"] > 0].reset_index(drop=True)
        res = simulate(cc, cfg)
        print(f"\n== {name} ==")
        for k in ("trades", "trades_per_month", "win_pct", "profit_factor", "total_return_pct", "cagr_pct",
                  "max_drawdown_pct", "losing_months_pct", "worst_month_pct", "longest_losing_streak",
                  "worst_day_pct", "skipped", "yearly_return_pct", "kill_switch_events"):
            print(f"  {k}: {res.get(k)}")
        out.append(dict(plan=name, **{k: res.get(k) for k in ("trades", "win_pct", "profit_factor", "total_return_pct",
                                                               "cagr_pct", "max_drawdown_pct")}))
    # per part (same replay, his plan)
    cc = c.copy()
    cc["mult"] = cc["kind"].map(PLANS["his plan 0.25 / 1 / 0.75"])
    print("\n== per part (trade level, before account rules) ==")
    for k, g in cc.groupby("kind"):
        print(f"  {k}: {len(g)} trades, win {100 * (g.r > 0).mean():.1f}% (gross R, costs added in the account replay)")
    pd.DataFrame(out).to_csv(ROOT / "data/research/confluence/three_part_sim.csv", index=False)


if __name__ == "__main__":
    main()
