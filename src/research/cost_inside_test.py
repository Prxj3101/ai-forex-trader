r"""
COST INSIDE SL / TP TEST (sirf research, live bot ko kuch nahi badalta).

Sawaal (Prajwal ka trading tarika):
  SL: charges SL ke ANDAR rakho -> lot thoda chhota, taaki SL + commission + slippage = poora risk ($25).
  TP: charges TP ke UPAR jodo  -> target thoda door, taaki TP hit pe charges ke baad bhi poora paisa mile.

Variants (same trades, same ML / filters / news block / risk rules):
  A  abhi jaisa         : lot = risk / SL,                 TP = 1R
  B  sirf SL andar      : lot = risk / (SL + charges),     TP = 1R
  C  sirf TP door       : lot = risk / SL,                 TP = 1R + commission   (TP pe net = SL ka paisa)
  D  dono (uska tarika) : lot = risk / (SL + charges),     TP = 1R + 2x commission + slippage
                          -> TP pe net jeet = SL pe net haar = poora risk (asli 1:1)
Trail wale trades (exit brain = trail) ka TP hota hi nahi -> unpe sirf lot size ka asar.

Chalana:  python src\research\cost_inside_test.py            (saare variants)
          python src\research\cost_inside_test.py --commission 7
Result:   data/research/cost_inside_test.json
"""
import argparse
import inspect
import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.risk import portfolio_sim as ps            # noqa: E402
from src.research.management import run_trade_full  # noqa: E402

CACHE = ps.RESEARCH / "cost_inside_cache"


def per_unit_costs(sym, entry, commission, slip_pips):
    """1 unit pe commission aur SL slippage (USD)."""
    slip_price = slip_pips * ps.PIP[sym]
    slip = slip_price if sym.endswith("USD") else slip_price / entry
    return commission / ps.LOT, slip


def candidates(tp_mode: str, commission: float, slip_pips: float) -> pd.DataFrame:
    """portfolio_sim.load_candidates jaisa, lekin tp1 trades ka target tp_mode ke hisaab se."""
    f = CACHE / f"cands_{tp_mode}_c{commission:g}_s{slip_pips:g}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    s = pd.read_parquet(ps.RESEARCH / "ml_scores.parquet")
    for c in ("symbol", "zone_tf", "direction"):
        s[c] = s[c].astype(str)
    s = s[(s["zone_tf"] == "M15") & (s["touch_number"] == 1) & s["keep_0.3"]]
    rows = []
    for sym, g in s.groupby("symbol"):
        m5 = pd.read_parquet(ps.DATA / f"{sym}_M5.parquet", columns=["high", "low", "close", "spread"])
        hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
        idx = m5.index
        for r in g.itertuples(index=False):
            f0 = int(idx.searchsorted(r.fill_time))
            g1 = min(f0 + ps.HOLD_BARS, len(m5))
            bull = r.direction == "bull"
            risk = abs(r.entry - r.sl)
            pol = ps.EXIT_POLICY[r.exit_ml]
            extra = 0.0
            if r.exit_ml == "tp1" and tp_mode != "base":
                pu = ps.risk_usd_per_unit(sym, r.entry, risk)
                comm_u, slip_u = per_unit_costs(sym, r.entry, commission, slip_pips)
                extra = comm_u / pu if tp_mode == "comm" else (2 * comm_u + slip_u) / pu
                pol = replace(pol, tp_r=1.0 + extra)
            rr, why, k = run_trade_full(pol, bull, r.entry, risk, hi[f0:g1], lo[f0:g1], cl[f0:g1], sp[f0:g1])
            rows.append(dict(symbol=sym, direction=r.direction, fill_time=r.fill_time,
                             exit_time=idx[f0 + k] + pd.Timedelta(minutes=5), entry=r.entry, risk=risk,
                             exit=r.exit_ml, r=rr, reason=why, score=r.score, tp_extra_r=extra,
                             va_pos=float(getattr(r, "r24_va_pos", np.nan)),
                             act=float(getattr(r, "act_15m", np.nan))))
    out = pd.DataFrame(rows).sort_values("fill_time").reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f)
    return out


def patch_sizing(inside: bool, slip_pips: float):
    """portfolio_sim.simulate ki lot-size line badalta hai (sirf is process mein)."""
    src = textwrap.dedent(inspect.getsource(ps._orig_simulate if hasattr(ps, "_orig_simulate") else ps.simulate))
    if not hasattr(ps, "_orig_simulate"):
        ps._orig_simulate = ps.simulate
    old = "units = (equity * risk_pct / 100) / per_unit"
    assert old in src, "portfolio_sim.simulate badal gaya - test update karo"
    if inside:
        new = ("_cu, _su = _COSTS(c.symbol, c.entry, cfg.commission_per_lot, cfg.slippage_pips)\n"
               "        units = (equity * risk_pct / 100) / (per_unit + _cu + _su)")
        src = src.replace(old, new)
    src = src.replace("def _orig_simulate", "def simulate")
    ns = dict(vars(ps))
    ns["_COSTS"] = per_unit_costs
    exec(compile(src, "simulate_patched", "exec"), ns)
    ps.simulate = ns["simulate"]




def run(commission: float, slip_pips: float) -> dict:
    variants = [("A", "abhi jaisa", False, "base"), ("B", "sirf SL andar", True, "base"),
                ("C", "sirf TP door (+commission)", False, "comm"), ("D", "dono (SL andar + TP door)", True, "both")]
    out = {}
    for key, label, inside, tp_mode in variants:
        cands = candidates(tp_mode, commission, slip_pips)
        patch_sizing(inside, slip_pips)
        res = ps.run(capital=10000, risk_pct=0.25, commission=commission, cands=cands, log=lambda *a, **k: None)
        tp1 = cands[cands["exit"] == "tp1"]
        out[key] = dict(label=label, sl_inside=inside, tp_mode=tp_mode,
                        tp1_trades=int(len(tp1)), tp1_hit_pct=round(100 * float((tp1["reason"] == "tp").mean()), 1),
                        avg_tp_extra_r=round(float(tp1["tp_extra_r"].mean()), 3),
                        **{k: res.get(k) for k in ("final_equity", "cagr_pct", "max_drawdown_pct", "profit_factor",
                                                   "win_pct", "trades", "trades_per_month", "worst_month_pct", "worst_day_pct",
                                                   "commission_usd", "avg_lots")},
                        yearly=res.get("yearly_return_pct"))
        print(f"{key} {label:28s} done", flush=True)
    ps.simulate = ps._orig_simulate
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission", type=float, default=5.0)
    ap.add_argument("--slip", type=float, default=0.2)
    ap.add_argument("--only-cache", default=None, help="sirf ek candidate set banao: base / comm / both")
    a = ap.parse_args()
    if a.only_cache:
        candidates(a.only_cache, a.commission, a.slip)
        print("cache ok", a.only_cache)
        raise SystemExit
    res = run(a.commission, a.slip)
    f = ps.RESEARCH / f"cost_inside_test_c{a.commission:g}.json"
    f.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in v if kk != "yearly"} for k, v in res.items()}, indent=1, default=str))
