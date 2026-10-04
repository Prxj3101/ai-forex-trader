r"""
LIVE STATUS: shadow bot ki data/live files padh ke dashboard ke liye ek JSON banata hai.

Do jagah use hota hai:
  - src/api/server.py   -> laptop dashboard (/api/live), jab bot isi computer pe chal raha ho
  - src/api/live_api.py -> VPS pe chhota read-only server, jahan bot asli mein chal raha hai

Sirf PADHTA hai, kabhi kuch likhta / badalta nahi.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_unix(ts) -> int | None:
    if ts is None or pd.isna(ts):
        return None
    return int((pd.Timestamp(ts) - EPOCH) // pd.Timedelta(seconds=1))


def clean(value):
    """NaN / numpy types ko JSON-friendly banao."""
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _tail_csv(f: Path, n: int, max_bytes: int = 400_000) -> pd.DataFrame:
    """Badi CSV ka sirf aakhri hissa padho (har 15 s poll pe poori file nahi)."""
    if not f.exists():
        return pd.DataFrame()
    with f.open("rb") as fh:
        header = fh.readline().decode("utf-8", "ignore")
        fh.seek(0, 2)
        size = fh.tell()
        start = max(len(header), size - max_bytes)
        fh.seek(start)
        chunk = fh.read().decode("utf-8", "ignore")
    if start > len(header):
        chunk = chunk.split("\n", 1)[1] if "\n" in chunk else ""
    try:
        d = pd.read_csv(io.StringIO(header + chunk))
    except Exception:  # noqa: BLE001 - adhi likhi line etc.
        return pd.DataFrame()
    return d.tail(n)


def _clean_rows(d: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in d.to_dict("records")]


def _pf(x: pd.Series):
    gl = -x[x < 0].sum()
    return round(float(x[x > 0].sum() / gl), 2) if gl > 0 else None


def build(live: Path, decisions: int = 40, trades: int = 30) -> dict:
    """Shadow bot ka live hisaab: account, khule trades, band trades, decisions, what-if summary."""
    from src.risk.portfolio_sim import RiskConfig, trade_risk_pct, trading_day

    out: dict = {"status": "none"}
    sf = live / "shadow_state.json"
    if not sf.exists():
        return out
    try:
        st = json.loads(sf.read_text())
    except Exception:  # noqa: BLE001 - file likhi ja rahi ho
        return out
    now = pd.Timestamp.now(tz="UTC")
    updated = pd.Timestamp(st.get("updated")) if st.get("updated") else None
    age = (now - updated).total_seconds() if updated is not None else None
    out["status"] = "running" if age is not None and age <= 600 else "stopped"
    out["updated"] = to_unix(updated) if updated is not None else None
    out["age_sec"] = round(age) if age is not None else None
    out["full_state"] = st.get("version") == 2

    cfg = RiskConfig(**{k: v for k, v in (st.get("risk") or {}).items() if k in RiskConfig.__dataclass_fields__})
    equity, peak = float(st.get("equity", cfg.capital)), float(st.get("peak", cfg.capital))
    month_start = float(st.get("month_start", equity))
    risk_now, braked = trade_risk_pct(cfg, equity, peak, month_start)
    today = str(trading_day(now))
    out["account"] = dict(
        capital=cfg.capital, equity=round(equity, 2), peak=round(peak, 2),
        dd_pct=round((peak - equity) / peak * 100, 2) if peak else 0.0,
        total_pct=round((equity / cfg.capital - 1) * 100, 2),
        today_pnl=round(float((st.get("day_pnl") or {}).get(today, 0.0)), 2),
        month_pnl=round(equity - month_start, 2), kill=bool(st.get("kill")), braked=bool(braked),
        risk_now_pct=round(risk_now, 3), brake_dd_pct=cfg.brake_dd_pct, kill_dd_pct=cfg.kill_dd_pct,
        daily_loss_pct=cfg.daily_loss_pct, daily_budget_pct=cfg.daily_risk_budget_pct,
    )
    out["open_trades"] = [
        {k: clean(t.get(k)) if not isinstance(t.get(k), str) else t.get(k)
         for k in ("symbol", "direction", "entry", "sl", "sl_pips", "lots", "risk_usd", "risk_pct", "exit", "score", "r_now")}
        | {"fill_time": to_unix(pd.Timestamp(t["fill_time"])) if t.get("fill_time") else None}
        for t in st.get("open_trades", [])
    ]

    ct = live / "closed_trades.csv"
    if ct.exists():
        c = pd.read_csv(ct)
        pnl = c["pnl"].astype(float) if "pnl" in c else pd.Series(dtype=float)
        r = c["r"].astype(float) if "r" in c else pd.Series(dtype=float)
        out["closed"] = dict(count=int(len(c)), wins=int((pnl > 0).sum()),
                             win_pct=round(float((pnl > 0).mean() * 100), 1) if len(c) else None,
                             pnl=round(float(pnl.sum()), 2), r_sum=round(float(r.sum()), 2), pf=_pf(pnl))
        c = c.assign(exit_ts=[to_unix(pd.Timestamp(x)) for x in c["exit_time"]],
                     fill_ts=[to_unix(pd.Timestamp(x)) for x in c["fill_time"]])
        eq = cfg.capital + pnl.cumsum()
        out["curve"] = [{"time": int(t), "value": round(float(v), 2)} for t, v in zip(c["exit_ts"], eq) if t]
        cols = [k for k in ("symbol", "direction", "fill_ts", "exit_ts", "entry", "exit", "exit_reason", "r", "lots", "pnl")
                if k in c]
        out["trades"] = _clean_rows(c[cols].tail(trades).iloc[::-1])
    else:
        out["closed"] = dict(count=0, wins=0, win_pct=None, pnl=0.0, r_sum=0.0, pf=None)
        out["curve"], out["trades"] = [], []

    d = _tail_csv(live / "decisions.csv", decisions)
    if len(d):
        d = d.assign(bar_ts=[to_unix(pd.Timestamp(x)) for x in d["bar_time"]])
        cols = [k for k in ("bar_ts", "symbol", "direction", "entry", "score", "decision", "reason", "lots", "exit")
                if k in d]
        out["decisions"] = _clean_rows(d[cols].iloc[::-1])
    else:
        out["decisions"] = []

    wf = live / "whatif.csv"
    if wf.exists():
        w = pd.read_csv(wf)
        rows = []
        for lab, g in w.groupby("decision"):
            rows.append(dict(decision=lab, zones=int(len(g)),
                             win_pct=round(float((g["net_brain"] > 0).mean() * 100), 1),
                             net_brain=round(float(g["net_brain"].sum()), 2), net_tp1=round(float(g["net_tp1"].sum()), 2),
                             net_trail=round(float(g["net_trail"].sum()), 2)))
        out["whatif"] = sorted(rows, key=lambda x: -x["zones"])
    else:
        out["whatif"] = []
    return out
