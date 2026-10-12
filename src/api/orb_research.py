"""
ORB research API (dashboard: Research -> ORB). Reads data/research/orb_research.parquet
(src/research/orb_research_data.py) + models/orb_brain.txt. Laptop only (downloaded data, not live).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "research" / "orb_research.parquet"
NY = "America/New_York"
STATUS_TEXT = {"traded": "TRADED", "skip_daily": "1/DAY LIMIT", "skip_ml": "ML SKIP", "skip_nocomp": "NO COMPRESSION",
               "no_break": "NO BREAK", "no_ml": "NO ML (pre-2019)"}
COMP = {"any": None, "nr4_or_tight": lambda d: (d.nr4 == 1) | (d.tight == 1), "nr4": lambda d: d.nr4 == 1,
        "nr7": lambda d: d.nr7 == 1, "tight": lambda d: d.tight == 1, "inside": lambda d: d.inside == 1,
        "none": lambda d: d.comp == 0}


@lru_cache(maxsize=2)
def _load(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(DATA)


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError("run: python src\\research\\orb_research_data.py")
    return _load(DATA.stat().st_mtime)


def _pf(r: pd.Series):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl > 0 else None


def _summ(r: pd.Series) -> dict:
    r = r.dropna()
    return dict(n=int(len(r)), pf=_pf(r), win=round(100 * float((r > 0).mean()), 1) if len(r) else None,
                r=round(float(r.sum()), 1), avg_r=round(float(r.mean()), 3) if len(r) else None)


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ----------------------------------------------------------------------
def report(pairs: list[str], ranges: list[str], comp: str = "nr4_or_tight", ml: bool = True, tp: str = "2R",
           one_per_day: bool = True, direction: str = "all", y0: int = 2016, y1: int = 2026) -> dict:
    d = load()
    col = f"BRK_{tp if tp in ('1R', '2R', 'EOD') else '2R'}_net"
    base = d[(d.broke == 1) & d.risk.notna() & d.symbol.isin(pairs) & d["range"].isin(ranges) & d.year.between(y0, y1)]
    if direction in ("bull", "bear"):
        base = base[base.direction == direction]
    x = base
    f = COMP.get(comp)
    if f is not None:
        x = x[f(x)]
    if ml:
        x = x[x.score.notna() & x.ml_keep.astype(bool)]
    if one_per_day:
        x = x.sort_values("break_ms").groupby(["symbol", "day"]).head(1)
    x = x.sort_values("break_ms")
    b = base.sort_values("break_ms").groupby(["symbol", "day"]).head(1) if one_per_day else base
    if ml:
        b = b[b.year >= 2019]
    years = sorted(x.year.unique().tolist())
    nyears = max(len(set(base.year if not ml else base[base.year >= 2019].year)), 1)
    eq = x[col].fillna(0).cumsum().to_numpy()
    pts = list(zip((x.break_ms // 1000).astype(int).tolist(), np.round(eq, 2).tolist()))
    if len(pts) > 1500:
        idx = np.linspace(0, len(pts) - 1, 1500).astype(int)
        pts = [pts[i] for i in idx]
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    by = lambda keys: [dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else (k,)), **_summ(g[col]))  # noqa: E731
                       for k, g in x.groupby(keys)]
    wd = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI"}
    by_wd = [dict(weekday=wd.get(int(k), str(k)), **_summ(g[col])) for k, g in x.groupby("weekday")] if "weekday" in x else []
    return dict(filters=dict(pairs=pairs, ranges=ranges, comp=comp, ml=ml, tp=tp, one_per_day=one_per_day, direction=direction,
                             years=[y0, y1], ml_years_note="ML on = only 2019-2026 (walk-forward: model never saw the year)" if ml else None),
                total=dict(_summ(x[col]), per_year=round(len(x) / nyears), max_dd_r=round(dd, 1),
                           pf_first_half=_pf(x[x.year <= (min(years) + max(years)) // 2][col].dropna()) if years else None,
                           pf_second_half=_pf(x[x.year > (min(years) + max(years)) // 2][col].dropna()) if years else None),
                baseline=dict(_summ(b[col]), label="same pairs / sessions / years, no compression, no ML"),
                by_year=by("year"), by_range_pair=by(["range", "symbol"]), by_symbol=by("symbol"), by_range=by("range"),
                by_weekday=by_wd, equity=pts)


# ----------------------------------------------------------------------
def _reason(r) -> str:
    parts = []
    if r["status"] == "skip_nocomp":
        return f"no compression: NR4 {'yes' if r['nr4'] == 1 else 'no'}, range {r['w_ratio']:.2f}x normal" if r["w_ratio"] == r["w_ratio"] else "no compression"
    parts.append(("NR4" if r["nr4"] == 1 else "") + (" + " if r["nr4"] == 1 and r["tight"] == 1 else "")
                 + (f"tight {r['w_ratio']:.2f}x" if r["tight"] == 1 else ""))
    if r["status"] == "no_break":
        return parts[0] + " · not broken before 11:00 NY"
    side = "BUY" if r["direction"] == "bull" else "SELL"
    bt = pd.Timestamp(int(r["break_ms"]), unit="ms", tz="UTC").tz_convert(NY).strftime("%H:%M")
    parts.append(f"{side} break {bt} NY")
    if r["score"] == r["score"]:
        parts.append(f"ML {r['score']:.2f} {'>=' if r['ml_keep'] else '<'} {r['thr']:.2f}")
    if r["status"] == "skip_daily":
        parts.append("pair already traded today")
    return " · ".join(p for p in parts if p)


def chart(symbol: str, start: int, end: int) -> list[dict]:
    """ORB ranges + trades of one pair between unix start / end (for the research history chart)."""
    from src.backtest import tick_engine as te
    d = load()
    x = d[(d.symbol == symbol) & (d.t1 >= start) & (d.t0 <= end)]
    out = []
    for _, r in x.iterrows():
        row = dict(day=r.day, symbol=r.symbol, range=r["range"], status=r.status, status_text=STATUS_TEXT.get(r.status, r.status),
                   reason=_reason(r), t0=int(r.t0), t1=int(r.t1), w_end=int(r.w_end), eod=int(r.eod), hi=_clean(r.hi),
                   lo=_clean(r.lo), w_ratio=_clean(r.w_ratio), nr4=_clean(r.nr4), tight=_clean(r.tight),
                   direction=r.direction if isinstance(r.direction, str) else None,
                   break_time=int(r.break_ms // 1000) if r.break_ms == r.break_ms else None,
                   entry=_clean(r.entry), score=_clean(r.score), open=False, sl=None, tp=None, r_now=None,
                   exit_time=None, r=None, r_net=None, exit_reason=None)
        if r.entry == r.entry and r.risk == r.risk and isinstance(r.direction, str):
            bull = r.direction == "bull"
            row["sl"] = _clean(r.entry - r.risk if bull else r.entry + r.risk)
            row["tp"] = _clean(r.entry + 2 * r.risk if bull else r.entry - 2 * r.risk)
        if r.status in ("traded", "skip_daily", "skip_ml", "no_ml") and row["sl"] is not None:
            row["r"], row["r_net"], row["exit_reason"] = _clean(r.BRK_2R), _clean(r.BRK_2R_net), r.BRK_2R_why
            if r.status == "traded":
                tk = te.ticks(symbol, int(r.break_ms), int(r.eod) * 1000)
                if tk is not None and len(tk[0]):
                    z = te.exit_arrays(*tk, r.direction == "bull", float(r.entry), float(row["sl"]), "tp2")
                    row["exit_time"] = int(z["exit_ms"] // 1000) if z["exit_ms"] else None
        out.append(row)
    return out


# ----------------------------------------------------------------------
def brain() -> dict:
    import lightgbm as lgb
    meta = json.loads((ROOT / "models/orb_meta.json").read_text())
    m = lgb.Booster(model_file=str(ROOT / "models/orb_brain.txt"))
    gain = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    gain = (gain / gain.sum()).sort_values(ascending=False)
    d = load()
    x = d[(d.broke == 1) & d.risk.notna() & d.score.notna()]
    col = "BRK_2R_net"
    x = x.assign(bucket=pd.qcut(x.score, 10, labels=False, duplicates="drop"))
    buckets = [dict(bucket=int(k) + 1, score_from=round(float(g.score.min()), 3), score_to=round(float(g.score.max()), 3),
                    **_summ(g[col])) for k, g in x.groupby("bucket")]
    keep = x[x.ml_keep.astype(bool)]
    return dict(meta={k: meta.get(k) for k in ("trained_at", "rows", "data_to", "threshold", "keep", "tp_r", "tight_q30", "label")},
                importance=[dict(feature=k, share=round(float(v), 4)) for k, v in gain.items()],
                buckets=buckets,
                split=dict(all_breaks=_summ(x[col]), ml_keep=_summ(keep[col]), ml_drop=_summ(x[~x.ml_keep.astype(bool)][col]),
                           ml_keep_comp=_summ(keep[keep.comp == 1][col]), ml_keep_nocomp=_summ(keep[keep.comp == 0][col]),
                           comp_no_ml=_summ(x[x.comp == 1][col])),
                note="Walk-forward 2019-2026: every score comes from a model trained only on earlier years. Net R at TP 2R.")
