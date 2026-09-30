"""
Research analysis: zone dataset pe filters lagao aur results nikalo (dashboard ke liye).

"What if" sawaal ka jawab deta hai, jaise:
    "Agar sirf wo zones lete jinka H1 displacement > 1.5 tha aur London mein fill hue, toh kya hota?"

Returns JSON-friendly dicts: overall stats (all vs filtered), breakdown tables, equity curves.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research"

BREAK_EVEN_WIN = 100 / 3   # 1:2 RR pe break-even win rate


def dataset_path(symbol: str, tf: str) -> Path:
    return OUT / f"{symbol}_{tf}_zones.parquet"


def load_dataset(symbol: str, tf: str) -> pd.DataFrame | None:
    p = dataset_path(symbol, tf)
    return pd.read_parquet(p) if p.exists() else None


def stats(d: pd.DataFrame) -> dict:
    """Sirf filled trades pe: kitne trades, win %, avg R, total R, profit factor."""
    t = d[d["outcome"] != "no_fill"]
    n = len(t)
    if n == 0:
        return {"zones": int(len(d)), "trades": 0, "win_pct": None, "avg_r": None, "total_r": 0.0, "pf": None}
    gross_win = float(t.loc[t["r_multiple"] > 0, "r_multiple"].sum())
    gross_loss = float(-t.loc[t["r_multiple"] < 0, "r_multiple"].sum())
    return {
        "zones": int(len(d)),
        "trades": int(n),
        "win_pct": round(100 * float((t["outcome"] == "win").mean()), 1),
        "avg_r": round(float(t["r_multiple"].mean()), 3),
        "total_r": round(float(t["r_multiple"].sum()), 1),
        "pf": round(gross_win / gross_loss, 2) if gross_loss else None,
    }


def apply_filters(d: pd.DataFrame, min_disp_h1: float = 0, sessions: list[str] | None = None,
                  fvg_only: bool = False, confluence_only: bool = False, direction: str = "both") -> pd.DataFrame:
    m = d["disp_h1"].fillna(0) >= min_disp_h1
    if sessions:
        # session_fill tabhi hota hai jab trade fill hua; no_fill zones ke liye created session dekho
        sess = d["session_fill"].fillna(d["session_created"])
        m &= sess.isin(sessions)
    if fvg_only:
        m &= d["has_fvg"]
    if confluence_only:
        m &= d["h1_confluence"]
    if direction in ("bull", "bear"):
        m &= d["direction"] == direction
    return d[m]


def equity_curve(d: pd.DataFrame, max_points: int = 400) -> list[dict]:
    """Trades ko time ke order mein jod ke cumulative R ki line (unique seconds, chart ke liye)."""
    t = d[d["outcome"] != "no_fill"].sort_values("fill_time")
    if t.empty:
        return []
    secs = ((t["fill_time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    cum = t["r_multiple"].cumsum().to_numpy()
    # same second pe do trades hon toh last value rakho (chart ko unique time chahiye)
    keep = np.r_[secs[1:] != secs[:-1], True]
    secs, cum = secs[keep], cum[keep]
    step = max(len(secs) // max_points, 1)
    idx = np.r_[np.arange(0, len(secs), step), len(secs) - 1]
    idx = np.unique(idx)
    return [{"time": int(secs[i]), "value": round(float(cum[i]), 2)} for i in idx]


def max_drawdown_r(d: pd.DataFrame) -> float:
    t = d[d["outcome"] != "no_fill"].sort_values("fill_time")
    if t.empty:
        return 0.0
    cum = t["r_multiple"].cumsum()
    return round(float((cum.cummax() - cum).max()), 1)


def breakdown(d: pd.DataFrame, col: str, order: list | None = None) -> list[dict]:
    rows = []
    for key, g in d.groupby(col, observed=True):
        rows.append({"label": str(key), **stats(g)})
    if order:
        pos = {str(k): i for i, k in enumerate(order)}
        rows.sort(key=lambda r: pos.get(r["label"], 99))
    return rows


def report(symbol: str, tf: str, **filters) -> dict | None:
    d = load_dataset(symbol, tf)
    if d is None:
        return None
    f = apply_filters(d, **filters)

    d = d.copy()
    f = f.copy()
    for x in (d, f):
        x["disp_bucket"] = pd.cut(x["disp_h1"], [0, 1, 1.5, 2.5, 100], labels=["<1", "1-1.5", "1.5-2.5", ">2.5"])
        x["session"] = x["session_fill"].fillna(x["session_created"])

    meta = d.iloc[0]
    return {
        "symbol": symbol,
        "tf": tf,
        "settings": {"swing_n": int(meta["swing_n"]), "min_disp": float(meta["min_disp"]), "rr": 2.0},
        "break_even_win_pct": round(BREAK_EVEN_WIN, 1),
        "all": {**stats(d), "max_dd_r": max_drawdown_r(d)},
        "filtered": {**stats(f), "max_dd_r": max_drawdown_r(f)},
        "equity_all": equity_curve(d),
        "equity_filtered": equity_curve(f),
        # Breakdowns FILTERED data pe (taaki filters ka asar har table mein dikhe)
        "by_session": breakdown(f, "session", ["Asian", "London", "London+NY", "New York", "Off-hours"]),
        "by_disp": breakdown(f, "disp_bucket", ["<1", "1-1.5", "1.5-2.5", ">2.5"]),
        "by_year": breakdown(f, "year"),
        "by_direction": breakdown(f, "direction", ["bull", "bear"]),
    }
