r"""
ORB RISK TEST (9 Oct 2026, his question): ORB trades with the SAME risk system as the zone bot (combo 0.5% / 0.25%,
house money, brake, kill switch, daily rules, position limits; src/risk/portfolio_sim.simulate) vs plain 0.25%.
Trades = walk-forward ORB (2019-26, model never saw the year, ML top 30% + compression, max 1 per pair per day),
gross R + exit time recomputed on ticks (TP 2R / SL / 16:00 NY). Costs inside simulate ($5/lot + 0.2 pip on stops).
Chalana: python src\research\orb_risk_test.py
"""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.live.orb import windows  # noqa: E402
from src.research import orb_ml_test as om  # noqa: E402
from src.research.orb_train import KEEP, tight_cutoffs  # noqa: E402
from src.risk import portfolio_sim as ps  # noqa: E402

CANDS = ROOT / "data/research/orb_risk_cands.parquet"


def candidates():
    if CANDS.exists():
        return pd.read_parquet(CANDS)
    d, res = om.wf(seeds=(1,))
    q = tight_cutoffs()
    tight = d.apply(lambda r: r.w_ratio <= q[r["range"]], axis=1)
    x = d[res[("2R", 1, KEEP)] & ((d.nr4 == 1) | tight) & (d.year >= 2019)]
    x = x.sort_values("break_ms").groupby(["symbol", "day"]).head(1)
    nr_rows = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/research/orb_nr").glob("nr_*.parquet"))])
    x = x.merge(nr_rows[["symbol", "day", "range", "BRK_entry", "BRK_risk"]], on=["symbol", "day", "range"], how="left")
    rows = []
    for r in x.itertuples(index=False):
        bull = r.direction == "bull"
        entry, risk = float(r.BRK_entry), float(r.BRK_risk)
        sl = entry - risk if bull else entry + risk
        eod = windows(str(r.day), r.range)[3]
        tk = te.ticks(r.symbol, int(r.break_ms), eod)
        z = te.exit_arrays(*tk, bull, entry, sl, "tp2")
        rows.append(dict(symbol=r.symbol, direction=r.direction, range=r.range,
                         fill_time=pd.Timestamp(int(r.break_ms), unit="ms", tz="UTC"),
                         exit_time=pd.Timestamp(int(z["exit_ms"]), unit="ms", tz="UTC"), entry=entry, risk=risk,
                         r=float(z["r"]), reason="time" if z["reason"] == "timeout" else z["reason"], exit="tp2",
                         r_net_bt=float(r.BRK_2R_net)))
    c = pd.DataFrame(rows).sort_values("fill_time").reset_index(drop=True)
    c.to_parquet(CANDS)
    return c


def variants():
    off = dict(brake_dd_pct=100.0, kill_dd_pct=100.0, daily_loss_pct=100.0, daily_risk_budget_pct=100.0,
               max_open=99, max_per_symbol=1, max_usd_same_side=99, house_money_pct=0.0)
    base = ps.RiskConfig(capital=10_000.0, commission_per_lot=5.0)
    combo = ps.combo_config(10_000.0, 5.0)
    return {
        "A  ORB now: fixed 0.25%, no extra rules": replace(base, risk_pct=0.25, risk_cap_pct=0.25, **off),
        "B  fixed 0.5%, no extra rules": replace(base, risk_pct=0.5, risk_cap_pct=0.5, **off),
        "C  ZONE BOT EXACT: combo 0.5/0.25 + house money + brake + kill + daily + limits": combo,
        "D  combo risk only (house money + brake + kill), no daily / position limits":
            replace(combo, daily_loss_pct=100.0, daily_risk_budget_pct=100.0, max_open=99, max_usd_same_side=99),
        "E  0.25% + house money (max 0.5%) + brake + kill + daily + limits (old zone setting)":
            replace(base, risk_pct=0.25, risk_cap_pct=0.5),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    c = candidates()
    print("ORB trades:", len(c), "| R check (gross tick vs backtest net):", round(float((c.r - c.r_net_bt).mean()), 3))
    rows, years = [], {}
    for name, cfg in variants().items():
        res = ps.simulate(c, cfg)
        rows.append(dict(setting=name, final=res["final_equity"], total_pct=res["total_return_pct"], cagr=res["cagr_pct"],
                         max_dd=res["max_drawdown_pct"], worst_day=res["worst_day_pct"], worst_month=res["worst_month_pct"],
                         trades=res["trades"], pf=res["profit_factor"], skipped=sum(res["skipped"].values()),
                         avg_risk=(res.get("risk_used") or {}).get("avg_pct"), kills=len(res["kill_switch_events"])))
        years[name[:2].strip()] = res["yearly_return_pct"]
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(years).to_string())
