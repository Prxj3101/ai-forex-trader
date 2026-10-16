"""
Priority test for the 3-part zone bot (16 Oct 2026). Same trades + account rules as three_part_sim.py,
his risk plan 0.25 / 1 / 0.75. Only the "same pair" rule changes:
  A  now      : 1 trade per pair, whoever comes first
  B  priority : a confluence / H4 trade may open even if a plain 15m trade is open on that pair (max 2 per pair);
                a plain trade still needs the pair empty
  C  B + no stacking: confluence and H4 never open together on the same pair + same direction (only the first)
  D  B + max open 4 (one extra slot only for confluence / H4)
Run: python src/research/priority_test.py
"""
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.risk import portfolio_sim as ps  # noqa: E402
from src.research import three_part_sim as tp  # noqa: E402

OLD_PAIR = 'if sum(p["symbol"] == c.symbol for p in open_pos) >= cfg.max_per_symbol:'
OLD_OPEN = "if len(open_pos) >= cfg.max_open:"


def make(rule):
    src = inspect.getsource(ps.simulate).replace(
        "risk_pct, braked = trade_risk_pct(cfg, equity, peak, month_start)",
        "risk_pct, braked = trade_risk_pct(cfg, equity, peak, month_start)\n        risk_pct = risk_pct * c.mult")
    src = src.replace("r=c.r, pnl=pnl,", "r=c.r, pnl=pnl, kind=c.kind,")
    src = src.replace(OLD_PAIR, "if PAIR_BLOCK(c, open_pos):")
    src = src.replace(OLD_OPEN, "if OPEN_BLOCK(c, open_pos, cfg):")
    src = src.replace("    res = report(closed, curve, skipped, report_cfg, len(cands))",
                      "    res = report(closed, curve, skipped, report_cfg, len(cands))\n    res['_closed'] = closed")
    ns = dict(vars(ps))
    ns["PAIR_BLOCK"], ns["OPEN_BLOCK"] = rule
    exec(src, ns)
    return ns["simulate"]


def pair_now(c, op):
    return any(p["symbol"] == c.symbol for p in op)


def pair_prio(c, op):
    same = [p for p in op if p["symbol"] == c.symbol]
    if c.kind == "plain":
        return len(same) > 0
    return len(same) >= 2 or any(p["kind"] != "plain" for p in same)


def pair_prio_nostack(c, op):
    if pair_prio(c, op):
        return True
    return c.kind != "plain" and any(p["symbol"] == c.symbol and p["direction"] == c.direction
                                     and p["kind"] != "plain" for p in op)


def open3(c, op, cfg):
    return len(op) >= cfg.max_open


def open4(c, op, cfg):
    return len(op) >= (cfg.max_open + 1 if c.kind != "plain" else cfg.max_open)


RULES = {"A now (1 per pair)": (pair_now, open3), "B priority (conf/H4 may join a plain)": (pair_prio, open3),
         "C priority + no conf/H4 stacking": (pair_prio_nostack, open3), "D priority + 4th slot for conf/H4": (pair_prio, open4)}


def main():
    c = tp.cands()
    c["mult"] = c["kind"].map(tp.PLANS["his plan 0.25 / 1 / 0.75"])
    cfg = replace(ps.RiskConfig(), capital=10000, risk_pct=1.0, risk_cap_pct=1.0, house_money_pct=0.0,
                  daily_loss_pct=2.0, daily_risk_budget_pct=2.5, kill_dd_pct=10.0, commission_per_lot=5.0)
    rows = []
    for name, rule in RULES.items():
        res = make(rule)(c, cfg)
        t = pd.DataFrame(res["_closed"])
        row = dict(rule=name, trades=res["trades"], win_pct=res["win_pct"], pf=res["profit_factor"],
                   total_pct=res["total_return_pct"], cagr=res["cagr_pct"], max_dd=res["max_drawdown_pct"],
                   losing_months=res["losing_months_pct"], worst_day=res["worst_day_pct"])
        for k in ("plain", "conf", "h4"):
            g = t[t.kind == k]
            row[f"n_{k}"] = len(g)
            row[f"pnl%_{k}"] = round(100 * g.pnl.sum() / 10000, 1)
        row["years"] = res["yearly_return_pct"]
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(df.drop(columns="years").to_string(index=False))
    for r in rows:
        print(r["rule"], r["years"])
    print("\nall candidates:", c.kind.value_counts().to_dict())
    df.to_csv(ROOT / "data/research/confluence/priority_test.csv", index=False)


if __name__ == "__main__":
    main()
