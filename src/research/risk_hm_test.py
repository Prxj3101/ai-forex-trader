"""
16 Oct 2026 (his questions): 3-part zone bot, same trades + account rules as the live setup
(2 per pair, max 5 open, USD side 2, daily 2% / 2.5%, brake 5% x0.5, kill 10%).
  Q1: H4 zone 0.75% vs 1%.
  Q2: house money ON/OFF. ON = risk of every type x (1 + 5% of this month's profit / equity), capped at
      1.5x or 2x the type's base (normal 0.25 -> max 0.5, confluence 1 -> max 2, H4 0.75 -> max 1.5 for 2x).
Run: python src/research/risk_hm_test.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.risk import portfolio_sim as ps  # noqa: E402
from src.research import three_part_sim as tp, priority_test as pt  # noqa: E402


def two(c, op):
    return sum(p["symbol"] == c.symbol for p in op) >= 2


def main():
    c0 = tp.cands()
    rows = []
    for h4 in (0.75, 1.0):
        for hm, cap in ((0.0, 1.0), (5.0, 1.5), (5.0, 2.0)):
            c = c0.copy()
            c["mult"] = c["kind"].map({"plain": 0.25, "conf": 1.0, "h4": h4})
            cfg = replace(ps.RiskConfig(), capital=10000, risk_pct=1.0, risk_cap_pct=cap, house_money_pct=hm,
                          daily_loss_pct=2.0, daily_risk_budget_pct=2.5, kill_dd_pct=10.0, commission_per_lot=5.0,
                          max_open=5, max_usd_same_side=2)
            r = pt.make((two, pt.open3))(c, cfg)
            t = pd.DataFrame(r["_closed"])
            rows.append(dict(h4_risk=h4, house_money="OFF" if not hm else f"ON max {cap:g}x", trades=r["trades"],
                             win=r["win_pct"], pf=r["profit_factor"], total=r["total_return_pct"], cagr=r["cagr_pct"],
                             max_dd=r["max_drawdown_pct"], worst_month=r["worst_month_pct"], worst_day=r["worst_day_pct"],
                             kills=len(r["kill_switch_events"]), max_risk=round(float(t.risk_pct.max()), 2),
                             years=r["yearly_return_pct"]))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.drop(columns="years").to_string(index=False))
    for x in rows:
        print(x["h4_risk"], x["house_money"], x["years"])
    df.to_csv(ROOT / "data/research/confluence/risk_hm_test.csv", index=False)


if __name__ == "__main__":
    main()
