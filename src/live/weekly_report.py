r"""
WEEKEND REPORT (15 Oct 2026, his ask): har hafte ka hisaab - bot ne kya kiya, TUNE kya kiya, aur kiska faisla sahi tha.

  ZONE BOT   band trades (R, PF, $), aur har chhua hua zone (whatif.csv: dono exits ka R pata hota hai):
             - bot ne liya (taken)            -> R
             - tune AVOID kiya (avoided)      -> bot ko kitna R milta (minus = tune bachaya)
             - tune TAKE kiya (taken_manual)  -> R
             - ML LOW (bot ne chhoda)         -> lete toh kitna R
             - tera EXIT badlav (1:1 <-> TRAIL) -> tera exit vs brain ka exit, R ka farak
             - tera manual CLOSE              -> band karte waqt R vs trade apne exit se kitna deta
  ORB        band trades (R, PF, $), tere manual close.
  TOTAL      shuru se ab tak tere faislon ka total fayda / nuksaan (R mein).
Files: data/live/weekly/report_<date>.json + .txt. Bot (shadow.run_live) har Saturday (NY) khud likhta hai.
Chalana: python src\live\weekly_report.py [--days 7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LIVE = ROOT / "data" / "live"
NY = "America/New_York"


def _read(f: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(f) if f.exists() else pd.DataFrame()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _t(s):
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")


def _pf(r):
    r = pd.Series(r, dtype=float).dropna()
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 2) if gl > 0 else (None if not len(r) else float("inf") if r.sum() > 0 else None)


def _box(r) -> dict:
    r = pd.Series(r, dtype=float).dropna()
    return dict(n=int(len(r)), R=round(float(r.sum()), 2), pf=_pf(r), win_pct=round(100 * float((r > 0).mean()), 1) if len(r) else None)


def _trade_exit_changes(wi: pd.DataFrame, acts: pd.DataFrame, ct: pd.DataFrame) -> dict | None:
    """Exit switched DURING a trade: real R of the trade (your exit, closed_trades.csv) vs R the brain's exit gives
    on the same zone (what-if, gross R both)."""
    if acts.empty or "kind" not in acts or ct.empty or wi.empty or "exit_bot" not in wi:
        return None
    ch = acts[acts.kind == "exit_trade"].drop_duplicates(["symbol", "fill_time"], keep="last")
    yours, brain = [], []
    for c in ch.itertuples():
        t = ct[(ct.symbol == c.symbol) & (_t(ct.fill_time).astype("int64") // 10**9 == int(c.fill_time))]
        w = wi[(wi.symbol == c.symbol) & (_t(wi.fill_time).astype("int64") // 10**9 == int(c.fill_time))]
        if len(t) and len(w):
            eb = str(w.exit_bot.iloc[0])
            yours.append(float(t.r.iloc[0]))
            brain.append(float(w["r_trail" if eb == "trail" else "r_tp1"].iloc[0]))
    if not yours:
        return dict(n=int(len(ch)), matched=0)
    g = np.array(yours) - np.array(brain)
    return dict(n=int(len(ch)), matched=len(yours), R_yours=round(float(sum(yours)), 2), R_brain=round(float(sum(brain)), 2),
                R_gained=round(float(g.sum()), 2), right=int((g > 0).sum()), wrong=int((g < 0).sum()),
                note="exit switched during the trade: real R vs the brain's exit on the same zone (gross R)")


def _decisions(wi: pd.DataFrame, closes: pd.DataFrame) -> dict:
    out = {}
    if wi.empty or "decision" not in wi:
        return out
    if "user" not in wi:
        wi = wi.assign(user="", exit_bot=wi.get("exit_brain"), net_bot=wi.get("net_brain"))
    wi = wi.copy()
    wi["user"] = wi["user"].fillna("")
    wi["net_bot"] = wi["net_bot"].fillna(wi["net_brain"])
    out["bot_taken"] = _box(wi.loc[wi.decision == "taken", "net_brain"])
    av = wi[wi.decision == "avoided"]
    out["you_avoided"] = _box(av["net_bot"]) | dict(note="R the bot would have made; negative = your AVOID saved that much")
    tk = wi[wi.decision == "taken_manual"]
    out["you_taken"] = _box(tk["net_brain"]) | dict(note="R of the zones you TOOK (bot would have skipped them)")
    out["ml_low_skipped"] = _box(wi.loc[wi.decision == "ml_low", "net_bot"]) | dict(note="R if the bot had taken its ML-low zones")
    ex = wi[(wi.exit_bot.notna()) & (wi.exit_brain != wi.exit_bot)]
    gain = (ex["net_brain"] - ex["net_bot"]).astype(float)
    out["your_exit_changes"] = dict(n=int(len(ex)), R_yours=round(float(ex["net_brain"].sum()), 2),
                                    R_brain=round(float(ex["net_bot"].sum()), 2), R_gained=round(float(gain.sum()), 2),
                                    right=int((gain > 0).sum()), wrong=int((gain < 0).sum()),
                                    note="only zones that were touched (trade or what-if); + = your exit beat the brain")
    if not closes.empty and "fill_time" in wi:
        rows = []
        for c in closes.itertuples():
            m = wi[(wi.symbol == c.symbol) & (_t(wi.fill_time).astype("int64") // 10**9 == int(c.fill_time))]
            if len(m):
                rows.append(float(c.r_now) - float(m.net_brain.iloc[0]))
        out["your_closes"] = dict(n=int(len(closes)), matched=len(rows), R_gained=round(float(np.sum(rows)), 2) if rows else 0.0,
                                  note="R at your close minus R the trade's own exit gave (+ = closing helped)")
    return out


def build(live: Path = LIVE, days: int = 7, end: pd.Timestamp | None = None) -> dict:
    end = end or pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    ct, wi = _read(live / "closed_trades.csv"), _read(live / "whatif.csv")
    acts, orb = _read(live / "user_actions.csv"), _read(live / "orb_trades.csv")
    rep = dict(made_at=end.isoformat(timespec="seconds"), from_=start.isoformat(timespec="seconds"),
               to=end.isoformat(timespec="seconds"), days=days)
    wk = lambda d, col: d[(_t(d[col]) >= start) & (_t(d[col]) < end)] if not d.empty and col in d else d  # noqa: E731
    c_w = wk(ct, "exit_time")
    col = lambda d, c: d[c] if c in d else pd.Series(dtype=float)  # noqa: E731
    rep["zone_account"] = _box(col(c_w, "r")) | dict(pnl_usd=round(float(col(c_w, "pnl").sum()), 2))

    def by_type(d):                                   # 16 Oct 2026: normal / confluence / 4H zone trades
        zt = d["zone_type"].fillna("normal") if "zone_type" in d else pd.Series("normal", index=d.index)
        return {k: _box(col(d[zt == k], "r")) | dict(pnl_usd=round(float(col(d[zt == k], "pnl").sum()), 2))
                for k in ("normal", "confluence", "h4")}
    rep["zone_by_type_week"] = by_type(c_w) if not c_w.empty else {}
    rep["zone_by_type_all_time"] = by_type(ct) if not ct.empty else {}
    closes = acts[acts.kind == "close"] if "kind" in acts else pd.DataFrame()
    rep["zone_week"] = _decisions(wk(wi, "fill_time"), wk(closes, "at"))
    rep["zone_all_time"] = _decisions(wi, closes)
    rep["zone_week"]["your_trade_exit_changes"] = _trade_exit_changes(wi, wk(acts, "at"), ct)
    rep["zone_all_time"]["your_trade_exit_changes"] = _trade_exit_changes(wi, acts, ct)
    o_w = wk(orb, "exit_time")
    rep["orb_account"] = _box(col(o_w, "r")) | dict(pnl_usd=round(float(col(o_w, "pnl").sum()), 2),
                                                     manual_closes=int((col(o_w, "exit_reason") == "manual").sum()))
    rt = _read(live / "real_trades.csv")                     # REAL DEMO orders vs shadow (src/live/execution.py)

    def real_box(d):
        out = {}
        if d.empty or "strategy" not in d:
            return out
        d = d.copy()
        if "zone_type" in d:                          # zone bot split: zone_normal / zone_confluence / zone_h4
            d["strategy"] = np.where(d["strategy"] == "zone", "zone_" + d["zone_type"].fillna("normal").astype(str), d["strategy"])
        for st, g in d.groupby("strategy"):
            out[st] = dict(n=int(len(g)), real_R=round(float(g["real_r"].dropna().sum()), 2),
                           shadow_R=round(float(g["shadow_r"].dropna().sum()), 2), net_usd=round(float(g["net_usd"].dropna().sum()), 2),
                           avg_slip_pips=round(float(g["entry_slip_pips"].dropna().mean()), 2) if g["entry_slip_pips"].notna().any() else None)
        return out
    rep["real_week"] = real_box(wk(rt, "real_exit_time"))
    rep["real_all_time"] = real_box(rt)
    a_w = wk(acts, "at")
    rep["your_actions_week"] = a_w["kind"].value_counts().to_dict() if "kind" in a_w else {}
    rep["your_actions_total"] = acts["kind"].value_counts().to_dict() if "kind" in acts else {}
    return rep


def text(rep: dict) -> str:
    def line(name, b):
        if not b:
            return f"  {name:22s} -"
        return f"  {name:22s} {b.get('n', 0):4d} zones  R {b.get('R', 0):+7.2f}  PF {b.get('pf')}"
    L = [f"WEEKLY REPORT {rep['from_'][:10]} -> {rep['to'][:10]}"]
    za = rep["zone_account"]
    L.append(f"ZONE BOT account: {za['n']} closed trades, R {za['R']:+.2f}, PF {za['pf']}, ${za['pnl_usd']:+,.2f}")
    for k, nm in (("normal", "NORMAL 15m (0.25%)"), ("confluence", "CONFLUENCE (1%)"), ("h4", "4H ZONE (1%)")):
        b = (rep.get("zone_by_type_week") or {}).get(k)
        if b:
            L.append(f"  {nm:22s} {b['n']:4d} trades  R {b['R']:+7.2f}  PF {b['pf']}  ${b['pnl_usd']:+,.2f}")
    for scope in ("zone_week", "zone_all_time"):
        z = rep.get(scope) or {}
        L.append("This week (touched zones):" if scope == "zone_week" else "Since the start:")
        for k, nm in (("bot_taken", "bot took"), ("you_avoided", "you AVOIDED (bot R)"), ("you_taken", "you TOOK"),
                      ("ml_low_skipped", "ML low (skipped)")):
            L.append(line(nm, z.get(k)))
        e = z.get("your_exit_changes")
        if e:
            L.append(f"  your EXIT changes      {e['n']:4d}  yours R {e['R_yours']:+.2f} vs brain R {e['R_brain']:+.2f} "
                     f"-> {e['R_gained']:+.2f}R ({e['right']} right / {e['wrong']} wrong)")
        e2 = z.get("your_trade_exit_changes")
        if e2 and e2.get("matched"):
            L.append(f"  exit switched in trade {e2['matched']:4d}  yours R {e2['R_yours']:+.2f} vs brain R {e2['R_brain']:+.2f} "
                     f"-> {e2['R_gained']:+.2f}R ({e2['right']} right / {e2['wrong']} wrong)")
        c = z.get("your_closes")
        if c:
            L.append(f"  your manual CLOSEs     {c['n']:4d}  -> {c['R_gained']:+.2f}R vs letting them run")
    oa = rep["orb_account"]
    L.append(f"ORB account: {oa['n']} closed trades, R {oa['R']:+.2f}, PF {oa['pf']}, ${oa['pnl_usd']:+,.2f}, manual closes {oa['manual_closes']}")
    for st, v in (rep.get("real_week") or {}).items():
        L.append(f"REAL DEMO {st.upper()}: {v['n']} trades, real R {v['real_R']:+.2f} vs shadow R {v['shadow_R']:+.2f}, "
                 f"net ${v['net_usd']:+,.2f}, avg entry slippage {v['avg_slip_pips']} pips")
    L.append(f"Your actions this week: {rep['your_actions_week']}")
    return "\n".join(L)


def write(live: Path = LIVE, days: int = 7) -> Path:
    rep = build(live, days)
    d = live / "weekly"
    d.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(rep["to"]).tz_convert(NY).strftime("%Y-%m-%d")
    (d / f"report_{stamp}.json").write_text(json.dumps(rep, indent=1, default=str))
    (d / f"report_{stamp}.txt").write_text(text(rep))
    return d / f"report_{stamp}.json"


def maybe_write(live: Path = LIVE) -> Path | None:
    """Called by the live bot loop: on Saturday (NY), once per week."""
    now = pd.Timestamp.now(tz=NY)
    if now.weekday() != 5:
        return None
    f = live / "weekly" / f"report_{now:%Y-%m-%d}.json"
    if f.exists():
        return None
    return write(live)


def latest(live: Path = LIVE) -> dict | None:
    d = live / "weekly"
    fs = sorted(d.glob("report_*.json")) if d.exists() else []
    if not fs:
        return None
    rep = json.loads(fs[-1].read_text())
    rep["text"] = (fs[-1].with_suffix(".txt")).read_text() if fs[-1].with_suffix(".txt").exists() else text(rep)
    rep["history"] = [f.stem.replace("report_", "") for f in fs][-12:]
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dir", default=str(LIVE))
    a = ap.parse_args()
    p = write(Path(a.dir), a.days)
    print(p.with_suffix(".txt").read_text())
