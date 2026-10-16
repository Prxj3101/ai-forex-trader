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

NY = "America/New_York"

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


TF_RULE = {"M5": "5min", "M15": "15min", "H1": "1h"}


ZONE_TYPES = ("normal", "confluence", "h4")     # 16 Oct 2026: 3-part zone bot (src/live/shadow.py ZONE_RISK)


def _ztype(v) -> str:
    return v if isinstance(v, str) and v in ZONE_TYPES else "normal"


def _zone_type_stats(c: pd.DataFrame) -> list[dict]:
    """Closed shadow trades per zone type: trades, win %, PF ($), net $, R."""
    zt = c["zone_type"].map(_ztype) if "zone_type" in c else pd.Series("normal", index=c.index)
    rows = []
    for k in ZONE_TYPES:
        g = c[zt == k]
        pnl = g["pnl"].astype(float) if "pnl" in g else pd.Series(dtype=float)
        rows.append(dict(zone_type=k, trades=int(len(g)), win_pct=round(float((pnl > 0).mean() * 100), 1) if len(g) else None,
                         pf=_pf(pnl), pnl=round(float(pnl.sum()), 2),
                         r_sum=round(float(g["r"].astype(float).sum()), 2) if "r" in g and len(g) else 0.0))
    return rows


def _zone_risk_now(cfg, equity: float, peak: float) -> dict:
    from src.live.shadow import ZONE_RISK
    braked = peak > 0 and (peak - equity) / peak * 100 >= cfg.brake_dd_pct
    return {k: round(v * (cfg.brake_factor if braked else 1.0), 3) for k, v in ZONE_RISK.items()}


def build_chart(live: Path, symbol: str, tf: str = "M15", n: int = 300) -> dict:
    """Live chart: bot ke MT5 candles (data/live/chart_SYM.json) + zones jo bot dekh raha hai + is pair ke trades."""
    tf = tf if tf in TF_RULE else "M15"
    out: dict = {"symbol": symbol, "tf": tf, "candles": [], "zones": [], "open": [], "trades": []}
    f = live / f"chart_{symbol}.json"
    if not f.exists():
        return out
    try:
        raw = json.loads(f.read_text())
    except Exception:  # noqa: BLE001 - file likhi ja rahi ho
        return out
    bars = list(raw.get("bars") or [])
    m5 = None
    # chal rahi candle (health.json, har ~5 s) jodo -> chart live hilta hai
    fb = None
    for name in ("ticks.json", "health.json"):          # ticks.json har ~1 s, health.json purana backup
        hf = live / name
        if not bars or not hf.exists():
            continue
        try:
            cand = (json.loads(hf.read_text()).get("forming") or {}).get(symbol)
            if cand and (fb is None or cand[0] > fb[0]):
                fb = cand
        except Exception:  # noqa: BLE001
            pass
    if fb and fb[0] > bars[-1][0]:
        bars.append(fb)
        out["forming_time"] = fb[0]
    if bars:
        d = pd.DataFrame(bars, columns=["t", "open", "high", "low", "close"])
        d.index = pd.to_datetime(d["t"], unit="s", utc=True)
        d = d[~d.index.duplicated(keep="last")].sort_index().dropna()
        m5 = d
        if tf != "M5":
            # trading clock (NY 5 PM = din ka shuru) jaisa hi: M15 / H1 boundaries UTC pe bhi same hote hain
            d = d.resample(TF_RULE[tf], label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        d = d.tail(n)
        out["candles"] = [{"time": int(t.timestamp()), "open": float(r.open), "high": float(r.high),
                           "low": float(r.low), "close": float(r.close)} for t, r in d.iterrows()]
    out["bar_time"] = raw.get("bar_time")
    wl = live / "watchlist.json"
    if wl.exists():
        try:
            out["zones"] = (json.loads(wl.read_text()).get("symbols", {}).get(symbol) or {}).get("zones", [])
        except Exception:  # noqa: BLE001
            pass
    sf = live / "shadow_state.json"
    open_raw: list[dict] = []
    if sf.exists():
        try:
            st = json.loads(sf.read_text())
            open_raw = [t for t in st.get("open_trades", []) if t.get("symbol") == symbol]
            out["open"] = [{"direction": t.get("direction"), "entry": clean(t.get("entry")), "sl": clean(t.get("sl")),
                            "tp": clean(t.get("tp")), "stop_now": clean(t.get("stop_now")), "best_r": clean(t.get("best_r")),
                            "exit": t.get("exit"), "lots": clean(t.get("lots")), "r_now": clean(t.get("r_now")),
                            "fill_time": to_unix(pd.Timestamp(t["fill_time"])) if t.get("fill_time") else None,
                            "zone": to_unix(pd.Timestamp(t["zone"])) if t.get("zone") else None,
                            "zone_type": t.get("zone_type") or "normal"}
                           for t in st.get("open_trades", []) if t.get("symbol") == symbol]
        except Exception:  # noqa: BLE001
            pass
    ct = live / "closed_trades.csv"
    closed_raw: list[dict] = []
    if ct.exists():
        try:
            c = pd.read_csv(ct)
            c = c[c["symbol"] == symbol].tail(30)
            closed_raw = c.to_dict("records")
            out["trades"] = [{"direction": r.direction, "entry": clean(r.entry), "sl": clean(r.sl), "exit": r.exit,
                              "r": clean(r.r), "exit_reason": r.exit_reason, "pnl": clean(r.pnl),
                              "fill_ts": to_unix(pd.Timestamp(r.fill_time)), "exit_ts": to_unix(pd.Timestamp(r.exit_time)),
                              "zone_type": _ztype(getattr(r, "zone_type", None))}
                             for r in c.itertuples(index=False)]
        except Exception:  # noqa: BLE001
            pass
    # jin zones se trade hua: box chart pe rehta hai (trade band hone pe exit time tak, halka)
    tz = live / "traded_zones.json"
    if tz.exists():
        try:
            exits = {t["fill_ts"]: t["exit_ts"] for t in out["trades"]}
            first = out["candles"][0]["time"] if out["candles"] else 0
            out["traded_zones"] = [dict(z, exit_ts=exits.get(z.get("fill_time")))
                                   for z in json.loads(tz.read_text())
                                   if z.get("symbol") == symbol and (z.get("fill_time") or 0) >= first - 6 * 86400]
        except Exception:  # noqa: BLE001
            out["traded_zones"] = []
    else:
        out["traded_zones"] = []
    # backup: jis trade ka zone box save nahi hua (purane trades / file missing) -> zone candle se dobara banao
    try:
        have = {z.get("fill_time") for z in out["traded_zones"]}
        exits = {t["fill_ts"]: t["exit_ts"] for t in out["trades"]}
        for t in [*open_raw, *closed_raw]:
            if not t.get("fill_time") or not t.get("zone") or pd.isna(t.get("zone")):
                continue
            fts = to_unix(pd.Timestamp(t["fill_time"]))
            if fts in have:
                continue
            z = _zone_from_candle(m5, pd.Timestamp(t["zone"]), t)
            if z:
                out["traded_zones"].append(dict(z, fill_time=fts, exit_ts=exits.get(fts)))
                have.add(fts)
    except Exception:  # noqa: BLE001
        pass
    out["open"] = [dict(t, kind="zone") for t in out["open"]] + _orb_open_rows(live, symbol)
    ex = _exec_status(live)                                  # real DEMO entry price per open trade (chart line)
    if ex:
        lk = ex.get("links") or {}
        for t in out["open"]:
            r = (lk.get("orb") or {}).get(f"{symbol}|{t.get('fill_time')}") if t.get("kind") == "orb" \
                else (lk.get("zone") or {}).get(f"{symbol}|{t.get('zone')}|{t.get('direction')}")
            if r and r.get("real_entry") is not None:
                t["real_entry"] = r["real_entry"]
    out["trades"] = [dict(t, kind="zone") for t in out["trades"]] + _orb_closed_rows(live, symbol)
    out["orb"] = chart_orb(live, symbol)                 # ORB dabbe (src/live/orb.py)
    return out


ORB_COMMISSION_PER_LOT = 5.0                           # src/live/orb.py: $5 / lot round trip


def _orb_open_rows(live: Path, symbol: str | None = None) -> list[dict]:
    """Khule ORB trades (orb_state.json) - wahi shape jo zone trades ka (chart lines + right column + live P&L)."""
    f = live / "orb_state.json"
    if not f.exists():
        return []
    try:
        st = json.loads(f.read_text())
    except Exception:  # noqa: BLE001 - file likhi ja rahi ho
        return []
    rows = []
    for t in st.get("open", []):
        if symbol and t.get("symbol") != symbol:
            continue
        pip = 0.01 if str(t.get("symbol", "")).endswith("JPY") else 0.0001
        ft = to_unix(pd.Timestamp(t["fill_time"])) if t.get("fill_time") else None
        entry, sl = float(t["entry"]), float(t["sl"])
        rows.append({"kind": "orb", "range": t.get("range"), "symbol": t.get("symbol"), "direction": t.get("direction"),
                     "entry": clean(entry), "sl": clean(sl), "tp": clean(t.get("tp")), "lots": clean(t.get("lots")),
                     "exit": "tp2", "sl_pips": round(abs(entry - sl) / pip, 1), "score": clean(t.get("score")),
                     "commission_usd": round(float(t.get("lots") or 0) * ORB_COMMISSION_PER_LOT, 2),
                     "stop_now": clean(sl), "best_r": clean(t.get("best_r")), "r_now": clean(t.get("r_now")),
                     "risk_usd": clean(t.get("risk_usd")), "filled_at": ft, "fill_time": ft})
    return rows


def _orb_closed_rows(live: Path, symbol: str, n: int = 30) -> list[dict]:
    """Band ORB trades (orb_trades.csv) for the chart: entry / SL / TP lines + exit marker."""
    f = live / "orb_trades.csv"
    if not f.exists():
        return []
    try:
        c = pd.read_csv(f)
    except Exception:  # noqa: BLE001
        return []
    c = c[c["symbol"] == symbol].tail(n)
    out = []
    for r in c.itertuples(index=False):
        risk = abs(float(r.entry) - float(r.sl))
        sign = 1 if r.direction == "bull" else -1
        out.append({"kind": "orb", "range": r.range, "direction": r.direction, "entry": clean(r.entry), "sl": clean(r.sl),
                    "tp": clean(r.tp), "exit": "tp2", "r": clean(r.r), "exit_reason": r.exit_reason, "pnl": clean(r.pnl),
                    "exit_px": clean(float(r.entry) + sign * float(r.r) * risk), "lots": clean(r.lots),
                    "fill_ts": to_unix(pd.Timestamp(r.fill_time)), "exit_ts": to_unix(pd.Timestamp(r.exit_time))})
    return out


def build_ticks(live: Path) -> dict:
    """Halka live data (dashboard har 1 s mangta hai): bid/ask, forming candle, khule trades (live P&L ke liye)."""
    out: dict = {"updated": None, "age_sec": None, "prices": {}, "forming": {}, "open": []}
    tf = live / "ticks.json"
    if tf.exists():
        try:
            t = json.loads(tf.read_text())
            out.update(updated=t.get("updated"), prices=t.get("prices") or {}, forming=t.get("forming") or {})
            if out["updated"]:
                out["age_sec"] = round((pd.Timestamp.now(tz="UTC") - pd.Timestamp(out["updated"])).total_seconds(), 1)
        except Exception:  # noqa: BLE001 - file likhi ja rahi ho
            pass
    # manual buttons: kaun se zones avoid hain, kaun se CLOSE abhi pending (bot ne abhi lagaya nahi)
    from src.live import control
    ctl = control.load(live)
    out["avoid"] = [{"symbol": a["symbol"], "zone": a["zone"], "direction": a["direction"]} for a in ctl["avoid"]]
    out["take"] = [{"symbol": a["symbol"], "zone": a["zone"], "direction": a["direction"]} for a in ctl["take"]]
    out["exit_set"] = [{"symbol": a["symbol"], "zone": a["zone"], "direction": a["direction"], "exit": a["exit"]}
                       for a in ctl["exit"]]
    out["close_pending"] = []
    out["exit_pending"] = []
    out["orb_close_pending"] = []
    of = live / "orb_state.json"
    try:
        odone = set(json.loads(of.read_text()).get("close_done", [])) if of.exists() else set()
    except Exception:  # noqa: BLE001
        odone = set()
    out["orb_close_pending"] = [{"symbol": x["symbol"], "fill_time": x["fill_time"]}
                                for x in ctl["orb_close"] if x["id"] not in odone]
    sf = live / "shadow_state.json"
    if sf.exists():
        try:
            st = json.loads(sf.read_text())
            commission_per_lot = float((st.get("risk") or {}).get("commission_per_lot", 5.0))
            done = set(st.get("close_done", []))
            out["close_pending"] = [{"symbol": x["symbol"], "fill_time": x["fill_time"]}
                                    for x in ctl["close"] if x["id"] not in done]
            out["exit_pending"] = [{"symbol": x["symbol"], "fill_time": x["fill_time"], "exit": x["exit"]}
                                   for x in ctl["trade_exit"] if x["id"] not in done]
            out["open"] = [{"symbol": t.get("symbol"), "direction": t.get("direction"), "entry": clean(t.get("entry")),
                            "sl": clean(t.get("sl")), "lots": clean(t.get("lots")), "exit": t.get("exit"),
                            "exit_bot": t.get("exit_bot"), "user_exit": t.get("user_exit", ""),
                            "taken_manual": t.get("user") == "take",
                            "sl_pips": clean(t.get("sl_pips")), "score": clean(t.get("score")),
                            "commission_usd": round(float(t.get("lots") or 0) * commission_per_lot, 2),
                            "tp": clean(t.get("tp")), "stop_now": clean(t.get("stop_now")),
                            "best_r": clean(t.get("best_r")), "r_now": clean(t.get("r_now")),
                            "filled_at": to_unix(pd.Timestamp(t["filled_at"])) if t.get("filled_at") else None,
                            "fill_time": to_unix(pd.Timestamp(t["fill_time"])) if t.get("fill_time") else None,
                            "zone": to_unix(pd.Timestamp(t["zone"])) if t.get("zone") else None,
                            "zone_type": _ztype(t.get("zone_type"))}
                           for t in st.get("open_trades", [])]
        except Exception:  # noqa: BLE001
            pass
    out["open"] = [dict(t, kind="zone") for t in out["open"]] + _orb_open_rows(live)   # ORB trades bhi (alag account)
    out["exec"] = _exec_status(live)
    if out["exec"]:                                  # real DEMO order per open trade (shadow vs real)
        links = out["exec"].get("links") or {}
        for t in out["open"]:
            if t.get("kind") == "orb":
                k = f"{t['symbol']}|{t.get('fill_time')}"
                r = (links.get("orb") or {}).get(k)
            else:
                k = f"{t['symbol']}|{t.get('zone')}|{t.get('direction')}"
                r = (links.get("zone") or {}).get(k)
            if r:
                t["real"] = r
    return out


def _exec_status(live: Path) -> dict | None:
    """data/live/exec_status.json (src/live/execution.py): switches, DEMO account, real positions / orders, errors."""
    f = live / "exec_status.json"
    if not f.exists():
        return None
    try:
        st = json.loads(f.read_text())
        st["age_sec"] = round((pd.Timestamp.now(tz="UTC") - pd.Timestamp(st["updated"])).total_seconds(), 1)
        return st
    except Exception:  # noqa: BLE001
        return None


def build_real(live: Path, n: int = 30) -> dict:
    """REAL vs SHADOW (real_trades.csv): per strategy totals + last trades."""
    f = live / "real_trades.csv"
    if not f.exists():
        return {"strategies": {}, "trades": []}
    try:
        d = pd.read_csv(f)
    except Exception:  # noqa: BLE001
        return {"strategies": {}, "trades": []}
    out = {}
    for s_, g in d.groupby("strategy"):
        rr, sr = g["real_r"].dropna(), g["shadow_r"].dropna()
        out[s_] = dict(trades=int(len(g)), real_r=round(float(rr.sum()), 2), shadow_r=round(float(sr.sum()), 2),
                       r_gap=round(float((g["real_r"] - g["shadow_r"]).dropna().mean()), 3) if len(rr) and len(sr) else None,
                       avg_entry_slip_pips=round(float(g["entry_slip_pips"].dropna().mean()), 2) if g["entry_slip_pips"].notna().any() else None,
                       net_usd=round(float(g["net_usd"].dropna().sum()), 2), commission_usd=round(float(g["commission_usd"].dropna().sum()), 2),
                       wins=int((g["net_usd"] > 0).sum()))
    cols = ["strategy", "symbol", "direction", "shadow_entry", "real_entry", "entry_slip_pips", "shadow_r", "real_r",
            "real_exit_reason", "net_usd", "real_exit_time", "note"]
    tail = d.tail(n).iloc[::-1][cols].astype(object).where(lambda x: x.notna(), None).to_dict("records")
    return {"strategies": out, "trades": tail}


def _zone_from_candle(m5, zt: pd.Timestamp, t: dict) -> dict | None:
    """Zone box = M15 zone candle ka wick (top/bottom) + body. Candle chart data mein na ho toh entry..SL ka box."""
    zt = zt.tz_localize("UTC") if zt.tzinfo is None else zt.tz_convert("UTC")
    bull = t.get("direction") == "bull"
    entry, sl = float(t["entry"]), float(t["sl"])
    z = {"direction": t.get("direction"), "zone_candle_time": to_unix(zt), "created_at": to_unix(zt),
         "disp_atr": None, "has_fvg": False, "entry": entry}
    w = m5.loc[(m5.index >= zt) & (m5.index < zt + pd.Timedelta(minutes=15))] if m5 is not None else None
    if w is not None and len(w):
        o, c = float(w["open"].iloc[0]), float(w["close"].iloc[-1])
        return dict(z, top=float(w["high"].max()), bottom=float(w["low"].min()),
                    body_top=max(o, c), body_bottom=min(o, c))
    return dict(z, top=max(entry, sl), bottom=min(entry, sl),
                body_top=entry if bull else max(entry, sl), body_bottom=min(entry, sl) if bull else entry)


def build(live: Path, decisions: int = 40, trades: int = 30) -> dict:
    """Shadow bot ka live hisaab: account, khule trades, band trades, decisions, what-if summary."""
    from src.risk.portfolio_sim import RiskConfig, trade_risk_pct, trading_day

    from src.live.monitor import tail_lines

    out: dict = {"status": "none"}
    now = pd.Timestamp.now(tz="UTC")

    # bot ki sehat (src/live/monitor.py, har ~10 s) + bot window ka log
    health = None
    hf = live / "health.json"
    if hf.exists():
        try:
            health = json.loads(hf.read_text())
            health["age_sec"] = round((now - pd.Timestamp(health["updated"])).total_seconds())
            health["uptime_sec"] = round((now - pd.Timestamp(health["started_at"])).total_seconds())
        except Exception:  # noqa: BLE001 - file likhi ja rahi ho
            health = None
    out["health"] = health
    out["log"] = tail_lines(live / "bot.log", 150)
    wl = live / "watchlist.json"
    try:
        out["watch"] = json.loads(wl.read_text()) if wl.exists() else None
    except Exception:  # noqa: BLE001
        out["watch"] = None

    sf = live / "shadow_state.json"
    if not sf.exists():
        return out
    try:
        st = json.loads(sf.read_text())
    except Exception:  # noqa: BLE001 - file likhi ja rahi ho
        return out
    updated = pd.Timestamp(st.get("updated")) if st.get("updated") else None
    age = (now - updated).total_seconds() if updated is not None else None
    out["status"] = "running" if age is not None and age <= 600 else "stopped"
    if health is not None:
        # health har ~10 s likhi jaati hai -> bot process zinda hai ya nahi, ye isse pakka pata chalta hai
        if health["age_sec"] > 120:
            out["status"] = "stopped"
        elif not health.get("market_open"):
            out["status"] = "market_closed"
        else:
            out["status"] = "running"
    out["updated"] = to_unix(updated) if updated is not None else None
    out["age_sec"] = round(age) if age is not None else None
    out["full_state"] = st.get("version") == 2

    cfg = RiskConfig(**{k: v for k, v in (st.get("risk") or {}).items() if k in RiskConfig.__dataclass_fields__})
    equity, peak = float(st.get("equity", cfg.capital)), float(st.get("peak", cfg.capital))
    month_start = float(st.get("month_start", equity))
    risk_now, braked = trade_risk_pct(cfg, equity, peak, month_start)
    today = str(trading_day(now))
    open_state = st.get("open_trades", [])
    open_commission = sum(float(t.get("lots") or 0) * cfg.commission_per_lot for t in open_state)
    out["account"] = dict(
        capital=cfg.capital, equity=round(equity, 2), peak=round(peak, 2),
        dd_pct=round((peak - equity) / peak * 100, 2) if peak else 0.0,
        total_pct=round((equity / cfg.capital - 1) * 100, 2),
        total_usd=round(equity - cfg.capital, 2),
        drawdown_usd=round(max(peak - equity, 0.0), 2),
        commission_per_lot=round(cfg.commission_per_lot, 2),
        commission_total=round(open_commission, 2),
        commission_closed=0.0,
        commission_open=round(open_commission, 2),
        today_pnl=round(float((st.get("day_pnl") or {}).get(today, 0.0)), 2),
        month_pnl=round(equity - month_start, 2), kill=bool(st.get("kill")), braked=bool(braked),
        risk_now_pct=round(risk_now, 3), brake_dd_pct=cfg.brake_dd_pct, kill_dd_pct=cfg.kill_dd_pct,
        daily_loss_pct=cfg.daily_loss_pct, daily_budget_pct=cfg.daily_risk_budget_pct,
        combo_enabled=cfg.combo_enabled, risk_mode=st.get('risk_mode'),
        base_risk_pct=cfg.risk_pct, risk_cap_pct=cfg.risk_cap_pct,
        zone_risk=_zone_risk_now(cfg, equity, peak), max_open=cfg.max_open, max_per_symbol=cfg.max_per_symbol,
    )
    out["open_trades"] = [
        {k: clean(t.get(k)) if not isinstance(t.get(k), str) else t.get(k)
         for k in ("symbol", "direction", "entry", "sl", "sl_pips", "lots", "risk_usd", "risk_pct", "exit", "score", "r_now",
                   "zone_type")}
        | {"commission_usd": round(float(t.get("lots") or 0) * cfg.commission_per_lot, 2)}
        | {"fill_time": to_unix(pd.Timestamp(t["fill_time"])) if t.get("fill_time") else None}
        for t in open_state
    ]

    ct = live / "closed_trades.csv"
    # Older/local shadow runs may only have the combined snapshot. It still
    # contains the closed rows, so use it when the append-only file is absent.
    if not ct.exists() and (live / "shadow_trades.csv").exists():
        ct = live / "shadow_trades.csv"
    if ct.exists():
        c = pd.read_csv(ct)
        if "status" in c:
            c = c[c["status"].astype(str).str.lower() == "closed"].copy()
        pnl = c["pnl"].astype(float) if "pnl" in c else pd.Series(dtype=float)
        r = c["r"].astype(float) if "r" in c else pd.Series(dtype=float)
        lots = pd.to_numeric(c["lots"], errors="coerce").fillna(0.0) if "lots" in c else pd.Series(0.0, index=c.index)
        if "commission_usd" in c:
            commission = pd.to_numeric(c["commission_usd"], errors="coerce").fillna(lots * cfg.commission_per_lot)
        else:
            commission = lots * cfg.commission_per_lot
        closed_commission = float(commission.sum())
        c = c.assign(commission_usd=commission.round(2))
        out["account"].update(
            commission_total=round(closed_commission + open_commission, 2),
            commission_closed=round(closed_commission, 2),
            commission_open=round(open_commission, 2),
        )
        out["closed"] = dict(count=int(len(c)), wins=int((pnl > 0).sum()),
                             win_pct=round(float((pnl > 0).mean() * 100), 1) if len(c) else None,
                             pnl=round(float(pnl.sum()), 2), r_sum=round(float(r.sum()), 2), pf=_pf(pnl),
                             commission=round(closed_commission, 2))
        out["by_zone_type"] = _zone_type_stats(c)
        c = c.assign(exit_ts=[to_unix(pd.Timestamp(x)) for x in c["exit_time"]],
                     fill_ts=[to_unix(pd.Timestamp(x)) for x in c["fill_time"]])
        eq = cfg.capital + pnl.cumsum()
        out["curve"] = [{"time": int(t), "value": round(float(v), 2)} for t, v in zip(c["exit_ts"], eq) if t]
        if "zone_type" not in c:
            c = c.assign(zone_type="normal")
        c = c.assign(zone_type=c["zone_type"].map(_ztype))
        cols = [k for k in ("symbol", "direction", "fill_ts", "exit_ts", "entry", "exit", "exit_reason", "r", "lots", "commission_usd",
                            "pnl", "zone_type") if k in c]
        out["trades"] = _clean_rows(c[cols].tail(trades).iloc[::-1])
    else:
        out["closed"] = dict(count=0, wins=0, win_pct=None, pnl=0.0, r_sum=0.0, pf=None, commission=0.0)
        out["by_zone_type"] = _zone_type_stats(pd.DataFrame())
        out["curve"], out["trades"] = [], []

    d = _tail_csv(live / "decisions.csv", decisions)
    if len(d):
        d = d.assign(bar_ts=[to_unix(pd.Timestamp(x)) for x in d["bar_time"]])
        cols = [k for k in ("bar_ts", "symbol", "direction", "entry", "score", "decision", "reason", "lots", "exit", "zone_type")
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
    out["real"] = build_real(live)
    return out


# ----------------------------------------------------------------------
# ORB SHADOW (src/live/orb.py): data/live/orb_state.json + orb_trades.csv
# ----------------------------------------------------------------------
ORB_STATUS_TEXT = {"waiting": "WAITING", "forming": "FORMING", "armed": "ARMED", "skip_nocomp": "NO COMPRESSION",
                   "no_break": "NO BREAK", "skip_ml": "ML SKIP", "skip_daily": "1/DAY LIMIT", "skip_news": "NEWS BLOCK", "skip_lot": "LOT TOO SMALL",
                   "traded": "TRADED", "no_data": "NO DATA"}


def _chart_bars(live: Path, symbol: str) -> pd.DataFrame | None:
    """Bot ke M5 bars (chart_SYM.json) + chalti candle -> forming dabbe ka abhi tak ka high / low."""
    f = live / f"chart_{symbol}.json"
    if not f.exists():
        return None
    try:
        bars = list(json.loads(f.read_text()).get("bars") or [])
        tf_ = live / "ticks.json"
        if tf_.exists():
            fb = (json.loads(tf_.read_text()).get("forming") or {}).get(symbol)
            if fb and bars and fb[0] > bars[-1][0]:
                bars.append(fb)
        d = pd.DataFrame(bars, columns=["t", "open", "high", "low", "close"])
        return d
    except Exception:  # noqa: BLE001
        return None


def _orb_ranges(live: Path, st: dict, symbol: str | None = None) -> list[dict]:
    recs = list((st.get("recs") or {}).values())
    if symbol:
        recs = [r for r in recs if r.get("symbol") == symbol]
    bars_cache: dict = {}
    out = []
    for r in recs:
        hi, lo = r.get("hi"), r.get("lo")
        if r.get("status") == "forming":                 # abhi ban raha: chart candles se running high / low
            s = r["symbol"]
            if s not in bars_cache:
                bars_cache[s] = _chart_bars(live, s)
            d = bars_cache[s]
            if d is not None and len(d):
                m = (d.t * 1000 >= r["r_start"]) & (d.t * 1000 < r["r_end"])
                if m.any():
                    hi, lo = float(d.high[m].max()), float(d.low[m].min())
        trade = next((t for t in st.get("open", []) if t.get("day") == r.get("day") and t.get("symbol") == r.get("symbol")
                      and t.get("range") == r.get("range")), None)
        out.append(dict(day=r.get("day"), symbol=r.get("symbol"), range=r.get("range"), status=r.get("status"),
                        status_text=ORB_STATUS_TEXT.get(r.get("status"), str(r.get("status")).upper()),
                        reason=r.get("reason", ""), t0=int(r["r_start"] // 1000), t1=int(r["r_end"] // 1000),
                        w_end=int(r["w_end"] // 1000), eod=int(r["eod"] // 1000), hi=clean(hi), lo=clean(lo),
                        w_ratio=clean(r.get("w_ratio")), nr4=r.get("nr4"), tight=r.get("tight"),
                        direction=r.get("direction"), break_time=int(r["break_ms"] // 1000) if r.get("break_ms") else None,
                        entry=clean(r.get("entry")), sl=clean(r.get("sl")), score=clean(r.get("score")),
                        open=bool(trade), tp=clean(trade.get("tp")) if trade else None,
                        r_now=clean(trade.get("r_now")) if trade else None))
    order = {"ASIA": 0, "LON30": 1, "LON60": 2}
    out.sort(key=lambda x: (x["day"], x["symbol"], order.get(x["range"], 9)))
    return out


def _orb_live_stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {}

    def summ(g):
        r = g["r"].astype(float)
        return dict(n=int(len(g)), pf=_pf(r), win=round(100 * float((r > 0).mean()), 1),
                    r=round(float(r.sum()), 2), pnl=round(float(g["pnl"].sum()), 2))
    return dict(total=summ(tr),
                by_range_pair=[dict(range=rg, symbol=s, **summ(g)) for (rg, s), g in tr.groupby(["range", "symbol"])],
                by_range=[dict(range=rg, **summ(g)) for rg, g in tr.groupby("range")],
                by_symbol=[dict(symbol=s, **summ(g)) for s, g in tr.groupby("symbol")])


def build_orb(live: Path, trades: int = 50) -> dict:
    """ORB panel + stats page: account, aaj ke dabbe (har pair), khule / band ORB trades, live stats."""
    out: dict = {"status": "none"}
    f = live / "orb_state.json"
    if not f.exists():
        return out
    try:
        st = json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return out
    now = pd.Timestamp.now(tz="UTC")
    upd = pd.Timestamp(st["updated"]) if st.get("updated") else None
    cap, eq, peak = float(st.get("capital", 10000)), float(st.get("equity", 10000)), float(st.get("peak", 10000))
    out.update(status="ok", updated=to_unix(upd) if upd is not None else None,
               age_sec=round((now - upd).total_seconds()) if upd is not None else None,
               threshold=st.get("threshold"), tight_q30=st.get("tight_q30"),
               account=dict(capital=cap, equity=round(eq, 2), peak=round(peak, 2), pnl=round(eq - cap, 2),
                            total_pct=round((eq / cap - 1) * 100, 2), dd_pct=round((peak - eq) / peak * 100, 2) if peak else 0,
                            risk_pct=st.get("risk_pct"), n_closed=st.get("n_closed", 0)))
    out["open"] = [{k: clean(t.get(k)) for k in ("day", "symbol", "range", "direction", "entry", "sl", "tp", "lots",
                                                    "risk_usd", "score", "r_now", "fill_time")} for t in st.get("open", [])]
    ny_today = str(now.tz_convert("America/New_York").date())
    rngs = _orb_ranges(live, st)
    days = sorted({r["day"] for r in rngs})
    show = [d for d in days if d >= ny_today][:1] or days[-1:]       # aaj (ya agla din) ke dabbe
    out["ranges"] = [r for r in rngs if r["day"] in show]
    out["day"] = show[0] if show else None
    tf_ = live / "orb_trades.csv"
    tr = pd.read_csv(tf_) if tf_.exists() else pd.DataFrame()
    out["trades"] = _clean_rows(tr.tail(trades).iloc[::-1]) if len(tr) else []
    out["live_stats"] = _orb_live_stats(tr)
    if len(tr):
        out["equity_curve"] = [[int(pd.Timestamp(t).timestamp()), float(e)] for t, e in zip(tr.exit_time, tr.equity)]
    return out


def chart_orb(live: Path, symbol: str) -> list[dict]:
    f = live / "orb_state.json"
    if not f.exists():
        return []
    try:
        return _orb_ranges(live, json.loads(f.read_text()), symbol)
    except Exception:  # noqa: BLE001
        return []


# ----------------------------------------------------------------------
# REAL DEMO ACCOUNT (src/live/execution.py): full panel like the shadow / ORB accounts
# ----------------------------------------------------------------------
def _pf_usd(v: pd.Series):
    v = v.dropna()
    gl = -v[v < 0].sum()
    return round(float(v[v > 0].sum() / gl), 2) if gl > 0 else None


def _real_stats(g: pd.DataFrame, now: pd.Timestamp) -> dict:
    if g.empty:
        return dict(trades=0)
    net = g["net_usd"].astype(float)
    t = pd.to_datetime(g["real_exit_time"], utc=True, errors="coerce", format="mixed")
    curve = net.fillna(0).cumsum()
    dd = float((curve.cummax().clip(lower=0) - curve).max()) if len(curve) else 0.0
    day0 = now.tz_convert(NY).normalize().tz_convert("UTC")
    wk0 = day0 - pd.Timedelta(days=now.tz_convert(NY).weekday())
    mo0 = now.tz_convert(NY).replace(day=1).normalize().tz_convert("UTC")
    rr, sr = g["real_r"].astype(float), g["shadow_r"].astype(float)
    both = (rr - sr).dropna()
    return dict(trades=int(len(g)), wins=int((net > 0).sum()), win_pct=round(100 * float((net > 0).mean()), 1),
                net_usd=round(float(net.sum()), 2), pf_usd=_pf_usd(net),
                gross_profit=round(float(g["profit_usd"].astype(float).sum()), 2),
                commission_usd=round(float(g["commission_usd"].astype(float).sum()), 2),
                swap_usd=round(float(g["swap_usd"].astype(float).fillna(0).sum()), 2),
                avg_win=round(float(net[net > 0].mean()), 2) if (net > 0).any() else None,
                avg_loss=round(float(net[net < 0].mean()), 2) if (net < 0).any() else None,
                best=round(float(net.max()), 2), worst=round(float(net.min()), 2), max_dd_usd=round(dd, 2),
                today=round(float(net[t >= day0].sum()), 2), week=round(float(net[t >= wk0].sum()), 2),
                month=round(float(net[t >= mo0].sum()), 2),
                real_r=round(float(rr.sum()), 2), shadow_r=round(float(sr.sum()), 2),
                r_gap_per_trade=round(float(both.mean()), 3) if len(both) else None,
                avg_slip_pips=round(float(g["entry_slip_pips"].astype(float).mean()), 2) if g["entry_slip_pips"].notna().any() else None,
                worst_slip_pips=round(float(g["entry_slip_pips"].astype(float).max()), 2) if g["entry_slip_pips"].notna().any() else None)


def build_real_account(live: Path, n: int = 150) -> dict:
    """Everything about the real DEMO account: status, account, open positions (with their shadow trade),
    pending orders, closed trades, stats (all / zone / orb), P&L curve, daily P&L, errors."""
    now = pd.Timestamp.now(tz="UTC")
    ex = _exec_status(live)
    out: dict = {"status": "ok" if ex else "none", "exec": ex}
    st = {}
    try:
        f = live / "exec_state.json"
        st = json.loads(f.read_text()) if f.exists() else {}
    except Exception:  # noqa: BLE001
        st = {}
    by_pos = {}
    for strat in ("zone", "orb"):
        for k, r in (st.get(strat) or {}).items():
            if r.get("position") is not None:
                by_pos[int(r["position"])] = dict(r, strategy=strat, key=k)
    shadow_open = {}
    try:
        ss = json.loads((live / "shadow_state.json").read_text()) if (live / "shadow_state.json").exists() else {}
        for t in ss.get("open_trades", []):
            shadow_open[f"{t['symbol']}|{to_unix(pd.Timestamp(t['zone']))}|{t['direction']}"] = t
    except Exception:  # noqa: BLE001
        pass
    opn = []
    for p in (ex or {}).get("positions", []):
        r = by_pos.get(int(p["ticket"]), {})
        sh = shadow_open.get(r.get("key", ""), {})
        risk = r.get("risk")
        now_px = p.get("price_now")
        r_now = (((now_px - p["price_open"]) if p["buy"] else (p["price_open"] - now_px)) / risk) if (now_px and risk) else None
        opn.append(dict(p, shadow_entry=r.get("shadow_entry"), slip_pips=r.get("entry_slip_pips"), risk=risk,
                        r_now=round(r_now, 3) if r_now is not None else None, opened=r.get("real_open_time"),
                        exit=sh.get("exit") if p["strategy"] == "zone" else "tp2", user_exit=sh.get("user_exit", ""),
                        taken_manual=sh.get("user") == "take", key=r.get("key"),
                        zone_type=_ztype(r.get("zone_type") or sh.get("zone_type")) if p["strategy"] == "zone" else None,
                        shadow_fill_time=(to_unix(pd.Timestamp(sh["fill_time"])) if sh.get("fill_time") else None)
                        if p["strategy"] == "zone" else (int(r["key"].split("|")[1]) if r.get("key") else None)))
    out["open"] = opn
    out["pending"] = (ex or {}).get("orders", [])
    f = live / "real_trades.csv"
    d = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["strategy"])
    if len(d):
        d["t_exit"] = pd.to_datetime(d["real_exit_time"], utc=True, errors="coerce", format="mixed")
        d = d.sort_values("t_exit")
    out["stats"] = {"all": _real_stats(d, now)} | {s_: _real_stats(g, now) for s_, g in d.groupby("strategy")} if len(d) \
        else {"all": dict(trades=0)}
    # 16 Oct 2026: zone bot split by zone type (normal 0.25% / confluence 1% / 4H 1%)
    zd = d[d["strategy"] == "zone"] if len(d) else d
    zt = zd["zone_type"].map(_ztype) if "zone_type" in zd else pd.Series("normal", index=zd.index)
    out["stats_zone_type"] = {k: _real_stats(zd[zt == k], now) for k in ZONE_TYPES}
    curve = []
    if len(d):
        cum = {"all": 0.0, "zone": 0.0, "orb": 0.0}
        for r in d.itertuples():
            v = float(r.net_usd) if pd.notna(r.net_usd) else 0.0
            cum["all"] += v
            cum[r.strategy] = cum.get(r.strategy, 0.0) + v
            if pd.notna(r.t_exit):
                curve.append(dict(t=int(r.t_exit.timestamp()), all=round(cum["all"], 2), zone=round(cum["zone"], 2),
                                  orb=round(cum["orb"], 2)))
        dd = d.dropna(subset=["t_exit"]).copy()
        dd["day"] = dd["t_exit"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
        daily = dd.groupby(["day", "strategy"])["net_usd"].sum().unstack(fill_value=0.0).tail(30)
        out["daily"] = [dict(day=i, **{c: round(float(v), 2) for c, v in row.items()}) for i, row in daily.iterrows()]
    else:
        out["daily"] = []
    out["curve"] = curve
    cols = [c for c in d.columns if c != "t_exit"]
    out["closed"] = d.tail(n).iloc[::-1][cols].astype(object).where(lambda x: x.notna(), None).to_dict("records") if len(d) else []
    return out
