"""
ZONE BOT 3-PART ACCOUNT REPORT for the dashboard RESEARCH tab (16 Oct 2026).
Same trades + rules as the live bot (src/live/shadow.py ZONE_RISK / zone_risk_config):
  NORMAL 15m 0.25% | CONFLUENCE (15m inside an alive H4 zone) 1% | 4H ZONE (h4_brain, 1:1) 1%,
  fixed risk, max 5 open, 2 per pair, 2 USD same side, daily 2% / 2.5%, brake 5% x0.5, kill 10%.
Trades = out-of-sample walk-forward 2019-26 (confluence/wf_base_h4flag.parquet + zone_h4/wf_n5.parquet), M5 labels.
Output: data/research/zone3_sim.json (portfolio_sim report format + "zone3" block). /api/risk/report serves it.
Run: python src/research/zone3_report.py
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.risk import portfolio_sim as ps  # noqa: E402
from src.research import three_part_sim as tp, priority_test as pt  # noqa: E402
from src.live.shadow import ZONE_RISK, zone_risk_config  # noqa: E402

OUT = ROOT / "data" / "research" / "zone3_sim.json"
KIND = {"plain": "normal", "conf": "confluence", "h4": "h4"}


def _pf(v):
    v = pd.Series(v, dtype=float)
    gl = -v[v < 0].sum()
    return round(float(v[v > 0].sum() / gl), 3) if gl > 0 else None


def run(c, mult, cfg, per_pair):
    c = c.copy()
    c["mult"] = c["kind"].map(mult)
    c = c[c["mult"] > 0].reset_index(drop=True)
    rule = (lambda x, op: sum(p["symbol"] == x.symbol for p in op) >= per_pair)
    return pt.make((rule, pt.open3))(c, cfg)


def main():
    c = tp.cands()
    live = zone_risk_config(10_000.0, 5.0)
    cfg = replace(live, risk_pct=1.0, risk_cap_pct=1.0)          # multiplier trick: risk = 1.0 x zone type risk
    mult = {k: ZONE_RISK[v] for k, v in KIND.items()}
    res = run(c, mult, cfg, live.max_per_symbol)
    t = pd.DataFrame(res.pop("_closed"))
    t["zone_type"] = t["kind"].map(KIND)
    parts = []
    for k in ("normal", "confluence", "h4"):
        g = t[t.zone_type == k]
        parts.append(dict(zone_type=k, risk_pct=ZONE_RISK[k], trades=int(len(g)),
                          win_pct=round(100 * float((g.pnl > 0).mean()), 1) if len(g) else None, pf=_pf(g.pnl),
                          net_usd=round(float(g.pnl.sum()), 2), r_sum=round(float(g.r.sum()), 1),
                          pf_19_21=_pf(g.loc[g.fill_time.dt.year <= 2021, "pnl"]),
                          pf_22_26=_pf(g.loc[g.fill_time.dt.year >= 2022, "pnl"])))
    # same data, old way (all 15m trades 0.5%, 1 per pair, max 3) -> what changed
    old_cfg = replace(cfg, max_open=3, max_per_symbol=1)
    old = run(c, {"plain": 0.5, "conf": 0.5, "h4": 0.0}, old_cfg, 1)
    old.pop("_closed")
    cmp_rows = [dict(name="Old: all 15m zones 0.5%, 1 per pair", **{k: old[k] for k in
                     ("trades", "win_pct", "profit_factor", "total_return_pct", "cagr_pct", "max_drawdown_pct")}),
                dict(name="Now: normal 0.25% / confluence 1% / 4H 1%, 2 per pair", **{k: res[k] for k in
                     ("trades", "win_pct", "profit_factor", "total_return_pct", "cagr_pct", "max_drawdown_pct")})]
    h4m = json.loads((ROOT / "models" / "h4_brain.json").read_text()) if (ROOT / "models" / "h4_brain.json").exists() else {}
    res["config"] = asdict(live)                                  # show the real live settings
    res["zone3"] = dict(zone_risk=ZONE_RISK, max_per_symbol=live.max_per_symbol, parts=parts, compare=cmp_rows,
                        h4_brain={k: h4m.get(k) for k in ("threshold", "trained_at", "h4_swing", "h4_life_days", "n_h4")},
                        note="Out-of-sample walk-forward 2019-26, M5 labels, no news block. Same rules as the live bot.",
                        built_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    res["account_profile"] = None
    OUT.write_text(json.dumps(res, default=str))
    print(f"saved {OUT.relative_to(ROOT)}: PF {res['profit_factor']}, win {res['win_pct']}%, total {res['total_return_pct']}%, "
          f"DD {res['max_drawdown_pct']}%")
    for p in parts:
        print(" ", p)
    for r in cmp_rows:
        print(" ", r)


if __name__ == "__main__":
    main()
