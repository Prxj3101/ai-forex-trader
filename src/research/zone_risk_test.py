r"""
ZONE BOT RISK TEST (9 Oct 2026, his question): current combo (0.5/0.25 + house money + brake + kill) vs other risk
settings, on the zone bot's out-of-sample trades 2019-26 (wick entry, tick exits; portfolio_sim.load_candidates,
cached in data/research/zone_risk_cands.parquet), after VA + activity filters (+ news rules if available).
All variants keep the bot's position limits and daily rules, so only the RISK SIZING changes.
Chalana: python src\research\zone_risk_test.py
"""
import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.risk import portfolio_sim as ps  # noqa: E402

CANDS = ROOT / "data/research/zone_risk_cands.parquet"


def candidates():
    c = pd.read_parquet(CANDS) if CANDS.exists() else ps.load_candidates()
    c = c[config.va_ok(c["va_pos"])]
    c = c[config.act_ok(c["act"])].reset_index(drop=True)
    try:
        from src.risk.news_rules import load_guard
        prof = config.profile()
        if prof.get("news_rules") or config.NEWS_BLOCK:
            c, _ = load_guard(prof["name"], log=lambda *a: None).apply_backtest(c, log=lambda *a: None)
    except Exception as e:  # noqa: BLE001
        print("news rules skipped:", e)
    return c.reset_index(drop=True)


def variants():
    base = ps.RiskConfig(capital=10_000.0, commission_per_lot=5.0, house_money_pct=0.0)
    combo = ps.combo_config(10_000.0, 5.0)
    return {
        "A  fixed 0.25%": replace(base, risk_pct=0.25, risk_cap_pct=0.25, brake_dd_pct=100.0),
        "B  fixed 0.4%": replace(base, risk_pct=0.4, risk_cap_pct=0.4, brake_dd_pct=100.0),
        "C  fixed 0.5%": replace(base, risk_pct=0.5, risk_cap_pct=0.5, brake_dd_pct=100.0),
        "D  CURRENT combo 0.5/0.25 + house money + brake": combo,
        "E  combo 0.5/0.25, NO house money": replace(combo, house_money_pct=0.0),
        "F  0.25% + house money (max 0.5%) + brake (old)": replace(base, risk_pct=0.25, risk_cap_pct=0.5, house_money_pct=5.0),
        "G  0.25% + brake only (half risk after 5% DD)": replace(base, risk_pct=0.25, risk_cap_pct=0.25),
        "H  0.4% + brake only": replace(base, risk_pct=0.4, risk_cap_pct=0.4),
    }


def run(c, variants_):
    rows, years = [], {}
    for name, cfg in variants_.items():
        res = ps.simulate(c, cfg)
        rows.append(dict(setting=name, final=res["final_equity"], total_pct=res["total_return_pct"], per_yr=res["cagr_pct"],
                         max_dd=res["max_drawdown_pct"], worst_month=res["worst_month_pct"], worst_day=res["worst_day_pct"],
                         trades=res["trades"], pf=res["profit_factor"], avg_risk=(res.get("risk_used") or {}).get("avg_pct"),
                         kills=len(res["kill_switch_events"])))
        years[name[:2].strip()] = res["yearly_return_pct"]
    return pd.DataFrame(rows), pd.DataFrame(years)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    c = candidates()
    print("zone bot trades after filters:", len(c))
    t, y = run(c, variants())
    print(t.to_string(index=False)); print(y.to_string())
