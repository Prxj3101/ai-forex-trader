"""
Research v2 (dashboard ka STRATEGY RESEARCH panel): dataset v2 pe "what if" filters, NET of costs.

v1 (analysis.py) se farak:
  - Dataset v2 (multi-touch, H1 zones, saare naye features), har pair ka displacement src/config.py se
  - Har result COMMISSION + SLIPPAGE ke baad (walkforward.add_costs wahi logic)
  - Exit choose karo: 1:1 / 1:2 / trailing / exit brain (ML ka choice)
  - "ML top 30%" toggle: sirf wo trades jo ML (out-of-sample) chunta. ML scores 2019 se hain.
  - Killzone + first-touch filters

Baseline ("ALL") = same pair + timeframe + exit, bina filters ke (ML on ho toh baseline bhi 2019+ period).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.ml import walkforward

RESEARCH = Path(__file__).resolve().parents[2] / "data" / "research"
EXITS = {"tp1": "Fixed 1:1", "tp2": "Fixed 1:2", "trail": "Trailing 1R after 1R", "brain": "Exit brain (ML picks 1:1 or trail)"}
BREAK_EVEN = {"tp1": 50.0, "tp2": 33.3}
KEYS = ["symbol", "zone_tf", "direction", "zone_candle_time", "touch_number"]


def _load(symbol: str, commission: float) -> pd.DataFrame | None:
    if not (RESEARCH / f"{symbol}_v2.parquet").exists():
        return None
    d = walkforward.load([symbol], commission)
    for c in ("symbol", "zone_tf", "direction", "killzone", "session_fill"):
        d[c] = d[c].astype(str)
    sc_file = RESEARCH / "ml_scores.parquet"
    if sc_file.exists():
        sc = pd.read_parquet(sc_file)
        for c in ("symbol", "zone_tf", "direction"):
            sc[c] = sc[c].astype(str)
        sc = sc[sc["symbol"] == symbol]
        cols = KEYS + ["score", "keep_0.3", "exit_ml"]
        d = d.merge(sc[[c for c in cols if c in sc]], on=KEYS, how="left")
    else:
        d["score"], d["keep_0.3"], d["exit_ml"] = np.nan, np.nan, None
    d["year"] = d["fill_time"].dt.year
    d["disp_bucket"] = pd.cut(d["disp_h1"], [0, 1, 1.5, 2.5, 100], labels=["<1", "1-1.5", "1.5-2.5", ">2.5"])
    return d


def _r(d: pd.DataFrame, exit_: str) -> pd.Series:
    if exit_ == "brain":
        return pd.Series(np.where(d["exit_ml"] == "trail", d["r_trail"], d["r_tp1"]), index=d.index)
    return d[{"tp1": "r_tp1", "tp2": "r_tp2", "trail": "r_trail"}[exit_]]


def _stats(r: pd.Series) -> dict:
    n = len(r)
    if n == 0:
        return {"zones": 0, "trades": 0, "win_pct": None, "avg_r": None, "total_r": 0.0, "pf": None, "max_dd_r": 0.0}
    gl = float(-r[r < 0].sum())
    cum = r.cumsum()
    return {"zones": n, "trades": n, "win_pct": round(100 * float((r > 0).mean()), 1),
            "avg_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 1),
            "pf": round(float(r[r > 0].sum()) / gl, 2) if gl else None,
            "max_dd_r": round(float((cum.cummax() - cum).max()), 1)}


def _equity(d: pd.DataFrame, r: pd.Series, max_points: int = 400) -> list[dict]:
    if d.empty:
        return []
    order = d["fill_time"].argsort(kind="stable")
    secs = ((d["fill_time"].iloc[order] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    cum = r.iloc[order].cumsum().to_numpy()
    keep = np.r_[secs[1:] != secs[:-1], True]
    secs, cum = secs[keep], cum[keep]
    step = max(len(secs) // max_points, 1)
    idx = np.unique(np.r_[np.arange(0, len(secs), step), len(secs) - 1])
    return [{"time": int(secs[i]), "value": round(float(cum[i]), 2)} for i in idx]


def _breakdown(d: pd.DataFrame, r: pd.Series, col: str, order: list | None = None) -> list[dict]:
    rows = [{"label": str(k), **_stats(r.loc[g.index])} for k, g in d.groupby(col, observed=True)]   # g time-order mein hai
    if order:
        pos = {str(k): i for i, k in enumerate(order)}
        rows.sort(key=lambda x: pos.get(x["label"], 99))
    return rows


def report(symbol: str, tf: str = "M15", exit_: str = "brain", ml_only: bool = True, first_touch: bool = True,
           min_disp_h1: float = 0, sessions: list[str] | None = None, killzones: list[str] | None = None,
           fvg_only: bool = False, confluence_only: bool = False, direction: str = "both",
           commission: float = walkforward.COMMISSION, va_filter: bool = False,
           act_filter: bool = False) -> dict | None:
    d = _load(symbol, commission)
    if d is None:
        return None
    swing_n, min_disp = int(d["swing_n"].iloc[0]), float(d["min_disp"].iloc[0])
    if tf in ("M15", "H1"):
        d = d[d["zone_tf"] == tf]
    has_ml = d["keep_0.3"].notna()
    ml_needed = ml_only or exit_ == "brain"
    if ml_needed:
        d = d[has_ml]                                     # ML scores sirf 2019+ (walk-forward test years)
    d = d.sort_values("fill_time")

    m = pd.Series(True, index=d.index)
    if ml_only:
        m &= d["keep_0.3"].astype(bool)
    if va_filter:                                         # value area filter: "mehengi" entries skip
        m &= d["va_ok"].astype(bool)
    if act_filter:                                        # activity filter: shaant market mein touch skip
        m &= d["act_ok"].astype(bool)
    if first_touch:
        m &= d["touch_number"] == 1
    m &= d["disp_h1"].fillna(0) >= min_disp_h1
    if sessions:
        m &= d["session_fill"].isin(sessions)
    if killzones:
        m &= d["killzone"].isin(killzones)
    if fvg_only:
        m &= d["has_fvg"].astype(bool)
    if confluence_only:
        m &= d["h1_confluence"].astype(bool)
    if direction in ("bull", "bear"):
        m &= d["direction"] == direction
    f = d[m]

    r_all, r_f = _r(d, exit_), _r(f, exit_)
    return {
        "symbol": symbol, "tf": tf,
        "settings": {"swing_n": swing_n, "min_disp": min_disp,
                     "exit": exit_, "exit_label": EXITS[exit_], "commission": commission,
                     "slippage_pips": walkforward.SLIP_PIPS, "net": True,
                     "period": f"{d['year'].min()}-{d['year'].max()}" if len(d) else "no ML scores for this pair",
                     "ml_period_only": ml_needed, "va_filter": va_filter,
                     "va_max_pos": config.VA_MAX_POS, "act_filter": act_filter, "act_min": config.ACT_MIN},
        "break_even_win_pct": BREAK_EVEN.get(exit_),
        "all": _stats(r_all), "filtered": _stats(r_f),
        "equity_all": _equity(d, r_all), "equity_filtered": _equity(f, r_f),
        "by_session": _breakdown(f, r_f, "session_fill", ["Asian", "London", "London+NY", "New York", "Off-hours"]),
        "by_killzone": _breakdown(f, r_f, "killzone", ["Asia open", "London open", "NY open", "none"]),
        "by_disp": _breakdown(f, r_f, "disp_bucket", ["<1", "1-1.5", "1.5-2.5", ">2.5"]),
        "by_year": _breakdown(f, r_f, "year"),
        "by_direction": _breakdown(f, r_f, "direction", ["bull", "bear"]),
    }
