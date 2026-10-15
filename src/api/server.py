r"""
Dashboard ka backend (FastAPI).

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    uvicorn src.api.server:app --reload

Phir browser mein:
    http://localhost:8000/docs   -> saare API endpoints, wahin se test bhi kar sakte ho

Endpoints:
    GET  /api/health     -> server zinda hai?
    GET  /api/symbols    -> kaunse pairs aur timeframes ka data hai
    GET  /api/candles    -> chart ke liye candles
    GET  /api/zones      -> kisi window ke orderflow zones
    POST /api/scan       -> 10 saal ka zone scan chalao (agents live dikhenge)
    POST /api/ml/train   -> ML Brain walk-forward training (ML agent live dikhega)
    GET  /api/ml/report  -> ML ke test results (score buckets, saal-wise PF, features)
    GET  /api/ml/scores  -> chart window ke trades ka out-of-sample ML score
    GET  /api/risk/report-> Risk Agent ka realistic account simulation (equity curve, drawdown, yearly %)
    GET  /api/config     -> bot settings (har pair ka displacement, core pairs)
    POST /api/research/v2/build  -> ek pair ka dataset v2 dobara banao
    GET  /api/research/v2/report -> research panel (NET of costs, exit choice, ML toggle, filters)
    WS   /ws             -> live agent events (dashboard yahan se sunta hai)

Rule: ye server sirf data DIKHATA aur tests CHALATA hai. Trading logic src/features,
src/strategies waghera mein hi rehta hai.
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.events import bus
from src.features.smc import find_orderflow_zones, track_zones

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1", "W1"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server start: event bus ko batao kaunsa event loop use karna hai
    bus.loop = asyncio.get_running_loop()
    bus.emit("system", "Trading engine online", status="done")
    yield


app = FastAPI(title="AI Forex Trader API", lifespan=lifespan)

# CORS: React dev server (port 5173) ko is server (port 8000) se baat karne do
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
@lru_cache(maxsize=16)
def load(symbol: str, tf: str) -> pd.DataFrame:
    """Parquet file load karta hai. lru_cache = ek baar load hua toh memory mein rakho (fast)."""
    path = DATA / f"{symbol}_{tf}.parquet"
    if not path.exists():
        raise HTTPException(404, f"No data for {symbol} {tf}. Run build_timeframes.py first.")
    return pd.read_parquet(path)


def to_unix(ts) -> int | None:
    """Timestamp -> seconds since 1970 (chart library isi format mein time leti hai)."""
    if ts is None or pd.isna(ts):
        return None
    return int((pd.Timestamp(ts) - EPOCH) // pd.Timedelta(seconds=1))


def window(df: pd.DataFrame, start: str | None, end: str | None, limit: int) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df.tail(limit) if not start else df.head(limit)


def clean(value):
    """NaN / numpy types ko JSON-friendly banao."""
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def zones_to_json(z: pd.DataFrame) -> list[dict]:
    time_cols = ["zone_candle_time", "bos_time", "created_at", "first_touch_time", "invalidated_time"]
    out = []
    for row in z.to_dict("records"):
        item = {k: (to_unix(v) if k in time_cols else clean(v)) for k, v in row.items()}
        out.append(item)
    return out


# ----------------------------------------------------------------------
# REST endpoints
# ----------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.get("/api/config")
def bot_config():
    """Bot ki settings (src/config.py): har pair ka displacement, core pairs."""
    from src import config
    p = config.profile()
    return {"swing_n": config.SWING_N, "core_symbols": config.CORE_SYMBOLS,
            "disp_by_symbol": config.DISP_BY_SYMBOL, "default_disp": config.DEFAULT_DISP,
            "account_profile": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}}


@app.get("/api/symbols")
def symbols():
    found: dict[str, list[str]] = {}
    for f in sorted(DATA.glob("*.parquet")):
        symbol, tf = f.stem.split("_", 1)
        found.setdefault(symbol, []).append(tf)
    return {s: [tf for tf in TIMEFRAMES if tf in tfs] for s, tfs in found.items()}


@app.get("/api/candles")
def candles(symbol: str, tf: str = "M15", start: str | None = None, end: str | None = None, limit: int = 2000):
    df = window(load(symbol, tf), start, end, min(limit, 20000))
    return [
        {"time": to_unix(t), "open": round(o, 5), "high": round(h, 5), "low": round(l, 5),
         "close": round(c, 5), "spread": round(s, 5)}
        for t, o, h, l, c, s in zip(df.index, df["open"], df["high"], df["low"], df["close"], df["spread"])
    ]


@app.get("/api/zones")
def zones(symbol: str, tf: str = "M15", start: str | None = None, end: str | None = None,
          swing_n: int = 1, min_disp: float = 2.0, limit: int = 2000):
    df = load(symbol, tf)
    view = window(df, start, end, min(limit, 20000))
    if view.empty:
        return []
    # Detector ko thoda pehle ka data bhi chahiye (ATR + swings ke liye)
    ctx = df[(df.index >= view.index[0] - pd.Timedelta(days=5)) & (df.index <= view.index[-1])]
    z = find_orderflow_zones(ctx, swing_n=swing_n, min_disp_atr=min_disp)
    if z.empty:
        return []
    z = track_zones(z, df)
    z = z[(z["created_at"] >= view.index[0] - pd.Timedelta(days=1)) & (z["created_at"] <= view.index[-1])]
    return zones_to_json(z)


# ----------------------------------------------------------------------
# Zone scan job (background thread, agents live events bhejte hain)
# ----------------------------------------------------------------------
scan_lock = threading.Lock()


def run_scan(symbol: str, tf: str, swing_n: int, min_disp: float) -> None:
    try:
        bus.emit("data", f"Loading {symbol} {tf} history", progress=10)
        df = load(symbol, tf)
        years = f"{df.index[0].year}-{df.index[-1].year}"
        bus.emit("data", f"{len(df):,} candles loaded ({years})", status="done", progress=100)

        bus.emit("structure", f"Finding swings (swing_n={swing_n}) and BOS", progress=30)
        t0 = time.time()
        z = find_orderflow_zones(df, swing_n=swing_n, min_disp_atr=min_disp)
        bus.emit("structure", "Market structure mapped", status="done", progress=100)
        bus.emit("zone", f"{len(z):,} orderflow zones found (min displacement {min_disp}x ATR)", progress=60)

        z = track_zones(z, df)
        touched = z["first_touch_time"].notna()
        dead = z["invalidated_time"].notna()
        result = {
            "symbol": symbol, "tf": tf, "swing_n": swing_n, "min_disp": min_disp,
            "zones": int(len(z)),
            "bull": int((z["direction"] == "bull").sum()),
            "bear": int((z["direction"] == "bear").sum()),
            "touched_pct": round(100 * touched.mean(), 1) if len(z) else 0,
            "invalidated_pct": round(100 * dead.mean(), 1) if len(z) else 0,
            "avg_touches": round(float(z["touches"].mean()), 2) if len(z) else 0,
            "with_fvg_pct": round(100 * z["has_fvg"].mean(), 1) if len(z) else 0,
            "seconds": round(time.time() - t0, 1),
        }
        bus.emit("zone", f"Touches tracked: {result['touched_pct']}% of zones were revisited", status="done", progress=100)
        bus.emit("system", f"Scan complete: {symbol} {tf}", status="done", kind="result", data=result)
    except Exception as e:
        bus.emit("system", f"Scan failed: {e}", status="error")
    finally:
        scan_lock.release()


@app.post("/api/scan")
def scan(symbol: str = "USDJPY", tf: str = "M15", swing_n: int = 1, min_disp: float = 2.0):
    load(symbol, tf)   # file hai ya nahi, pehle check (warna 404)
    if not scan_lock.acquire(blocking=False):
        raise HTTPException(409, "A scan is already running")
    threading.Thread(target=run_scan, args=(symbol, tf, swing_n, min_disp), daemon=True).start()
    return {"started": True}


# ----------------------------------------------------------------------
# Research: zone dataset (Setup #1 ka 10 saal ka result) + filters
# ----------------------------------------------------------------------
from src.research import analysis, zone_dataset  # noqa: E402

research_lock = threading.Lock()


def run_research(symbol: str, tf: str, swing_n: int, min_disp: float) -> None:
    try:
        bus.emit("data", f"Building research dataset: {symbol} {tf}", progress=5)

        def log(agent: str, msg: str, pct=None):
            bus.emit(agent, msg, status="done" if pct == 100 else "active", progress=pct)

        ds = zone_dataset.build(symbol, tf, swing_n, min_disp, log=log)
        zone_dataset.save(ds, symbol, tf)
        s = analysis.stats(ds)
        bus.emit("system", f"Research ready: {s['trades']:,} trades, win {s['win_pct']}%, PF {s['pf']}",
                 status="done", kind="research", data={"symbol": symbol, "tf": tf, **s})
    except Exception as e:
        bus.emit("system", f"Research failed: {e}", status="error")
    finally:
        research_lock.release()


@app.post("/api/research/build")
def research_build(symbol: str = "USDJPY", tf: str = "M15", swing_n: int = 1, min_disp: float = 2.0):
    for t in (tf, "M5", "H1", "H4"):
        load(symbol, t)   # saari files hain ya nahi, pehle check
    if not research_lock.acquire(blocking=False):
        raise HTTPException(409, "A research build is already running")
    threading.Thread(target=run_research, args=(symbol, tf, swing_n, min_disp), daemon=True).start()
    return {"started": True}


@app.get("/api/research/report")
def research_report(symbol: str = "USDJPY", tf: str = "M15", min_disp_h1: float = 0, sessions: str = "",
                    fvg_only: bool = False, confluence_only: bool = False, direction: str = "both"):
    rep = analysis.report(
        symbol, tf, min_disp_h1=min_disp_h1,
        sessions=[s for s in sessions.split(",") if s], fvg_only=fvg_only,
        confluence_only=confluence_only, direction=direction,
    )
    if rep is None:
        raise HTTPException(404, f"No research dataset for {symbol} {tf}. Build it first.")
    return rep


# ----------------------------------------------------------------------
# Research v2: dataset v2, NET of costs, exit choice, ML toggle (dashboard STRATEGY RESEARCH panel)
# ----------------------------------------------------------------------
def run_research_v2(symbol: str) -> None:
    try:
        from src.research import zone_dataset_v2

        bus.emit("data", f"Building research dataset v2: {symbol}", progress=5)

        def log(agent: str, msg: str, pct=None):
            bus.emit(agent, msg, status="done" if pct == 100 else "active", progress=pct)

        ds = zone_dataset_v2.build(symbol, log=log)          # swing + displacement src/config.py se
        zone_dataset_v2.save(ds, symbol)
        bus.emit("system", f"Research v2 ready: {symbol}, {len(ds):,} trades. Retrain the ML Brain to update scores.",
                 status="done", kind="research", data={"symbol": symbol, "trades": len(ds)})
    except Exception as e:
        bus.emit("system", f"Research failed: {e}", status="error")
    finally:
        research_lock.release()


@app.post("/api/research/v2/build")
def research_v2_build(symbol: str = "USDJPY"):
    for t in ("M5", "M15", "H1", "H4", "D1"):
        load(symbol, t)
    if not research_lock.acquire(blocking=False):
        raise HTTPException(409, "A research build is already running")
    threading.Thread(target=run_research_v2, args=(symbol,), daemon=True).start()
    return {"started": True}


@app.get("/api/research/v2/report")
def research_v2_report(symbol: str = "USDJPY", tf: str = "M15", exit: str = "brain", ml_only: bool = True,
                       first_touch: bool = True, min_disp_h1: float = 0, sessions: str = "", killzones: str = "",
                       fvg_only: bool = False, confluence_only: bool = False, direction: str = "both",
                       va_filter: bool = False, act_filter: bool = False):
    from src.research import analysis_v2

    if exit not in analysis_v2.EXITS:
        raise HTTPException(400, f"exit must be one of {list(analysis_v2.EXITS)}")
    rep = analysis_v2.report(
        symbol, tf, exit_=exit, ml_only=ml_only, first_touch=first_touch, min_disp_h1=min_disp_h1,
        sessions=[s for s in sessions.split(",") if s], killzones=[k for k in killzones.split(",") if k],
        fvg_only=fvg_only, confluence_only=confluence_only, direction=direction, va_filter=va_filter,
        act_filter=act_filter,
    )
    if rep is None:
        raise HTTPException(404, f"No dataset v2 for {symbol}. Build it first.")
    return rep


# ----------------------------------------------------------------------
# ML Brain: walk-forward training + scores
# ----------------------------------------------------------------------
import json  # noqa: E402

RESEARCH = ROOT / "data" / "research"
ml_lock = threading.Lock()
SIM_CAPITAL = 10_000     # account simulation ka starting capital (USD)
SIM_RISK = 0.5           # Combo: 0.5% -> 0.25% at 3% DD, recover at 1% DD.


def run_ml_train(symbols: list[str]) -> None:
    try:
        from src.ml import walkforward   # yahan import: lightgbm na ho toh baaki server phir bhi chale

        bus.emit("ml", f"Training ML Brain on {', '.join(symbols)} (walk-forward)", progress=5)
        step = {"n": 0}

        def log(msg: str):
            step["n"] += 1
            bus.emit("ml", msg.strip(), progress=min(95, 5 + step["n"] * 10))

        res, _ = walkforward.train(symbols, log=log)
        allr = next(r for r in res["summary"] if r["keep"] == 1.0)
        bot = res["bot"]
        bus.emit("ml", f"Unseen years (net of costs): all zones PF {allr['pf_trail']} -> bot PF {bot['pf']}",
                 status="done", progress=100)

        # Risk Agent: naye ML ke trades pe realistic account simulation
        from src.risk import portfolio_sim
        bus.emit("risk", f"Simulating ${SIM_CAPITAL:,.0f} account ({SIM_RISK}% risk, ${walkforward.COMMISSION:g}/lot)", progress=30)
        sim = portfolio_sim.run(SIM_CAPITAL, SIM_RISK, walkforward.COMMISSION,
                                log=lambda m: bus.emit("risk", m.strip(), progress=60), combo=True)
        portfolio_sim.save(sim)
        bus.emit("risk", f"Account sim: {sim['cagr_pct']}%/yr, max drawdown {sim['max_drawdown_pct']}%, "
                         f"{sim['trades_per_month']} trades/month", status="done", progress=100)
        bus.emit("system", "ML Brain trained", status="done", kind="ml", data={"bot_pf": bot["pf"]})
    except Exception as e:
        bus.emit("ml", f"Training failed: {e}", status="error")
    finally:
        ml_lock.release()


@app.post("/api/ml/train")
def ml_train(symbols: str = "EURUSD,USDJPY,GBPUSD"):     # = src/config.py CORE_SYMBOLS
    syms = [s for s in symbols.split(",") if s]
    missing = [s for s in syms if not (RESEARCH / f"{s}_v2.parquet").exists()]
    if missing:
        raise HTTPException(404, f"Dataset v2 missing for {', '.join(missing)}. Run zone_dataset_v2.py first.")
    if not ml_lock.acquire(blocking=False):
        raise HTTPException(409, "ML training is already running")
    threading.Thread(target=run_ml_train, args=(syms,), daemon=True).start()
    return {"started": True}


@app.get("/api/ml/report")
def ml_report():
    f = RESEARCH / "ml_walkforward.json"
    if not f.exists():
        raise HTTPException(404, "ML Brain not trained yet")
    r = json.loads(f.read_text())
    imp = sorted(r.get("importance", {}).items(), key=lambda kv: -kv[1])[:10]
    return {
        "trained_at": r.get("trained_at"),
        "symbols": r.get("symbols", []),
        "n_trades": r.get("n_trades"),
        "n_features": len(r.get("features", [])),
        "first_test_year": r.get("first_test_year", 2019),
        "summary": r["summary"],
        "score_buckets": r["score_buckets"],
        "by_year": r["by_year"],
        "by_symbol": r["by_symbol"],
        "importance": [{"feature": k, "pct": v} for k, v in imp],
        "exits": r.get("exits", []),
        "bot": r.get("bot"),
        "va_filter": r.get("va_filter"),
        "act_filter": r.get("act_filter"),
        "commission": r.get("commission"),
        "slippage_pips": r.get("slippage_pips"),
        "net": r.get("results_are") == "net of costs",
    }


@app.get("/api/risk/report")
def risk_report():
    """Risk Agent ka account simulation (portfolio_sim.json)."""
    f = RESEARCH / "portfolio_sim.json"
    if not f.exists():
        raise HTTPException(404, "No account simulation yet. Retrain the ML Brain or run portfolio_sim.py.")
    return json.loads(f.read_text())


@lru_cache(maxsize=2)
def _ml_scores(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(RESEARCH / "ml_scores.parquet")


@app.get("/api/ml/scores")
def ml_scores(symbol: str, tf: str = "M15", start: str | None = None, end: str | None = None):
    """Chart window ke trades ka OUT-OF-SAMPLE ML score (walk-forward test se)."""
    f = RESEARCH / "ml_scores.parquet"
    if not f.exists():
        return []
    s = _ml_scores(f.stat().st_mtime)
    s = s[(s["symbol"] == symbol) & (s["zone_tf"] == tf)]
    if start:
        s = s[s["fill_time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        s = s[s["fill_time"] <= pd.Timestamp(end, tz="UTC")]
    return [
        {"fill_time": to_unix(r.fill_time), "zone_candle_time": to_unix(r.zone_candle_time),
         "direction": r.direction, "touch_number": int(r.touch_number), "entry": clean(r.entry), "sl": clean(r.sl),
         "score": round(float(r.score), 3), "keep_top30": bool(r.keep_0_3), "keep_top50": bool(r.keep_0_5),
         "r_trail": clean(r.r_trail), "exit": r.r_trail_exit,
         "exit_ml": getattr(r, "exit_ml", None), "r_ml": clean(getattr(r, "r_ml", np.nan)),
         "va_ok": bool(getattr(r, "va_ok", True)), "va_pos": clean(getattr(r, "r24_va_pos", np.nan)),
         "act_ok": bool(getattr(r, "act_ok", True)), "act": clean(getattr(r, "act_15m", np.nan))}
        for r in s.rename(columns={"keep_0.3": "keep_0_3", "keep_0.5": "keep_0_5"}).itertuples(index=False)
    ]


# ----------------------------------------------------------------------
# LIVE SHADOW BOT: data/live files (src/live/shadow.py likhta hai), dashboard sirf padhta hai
# ----------------------------------------------------------------------
# Bot agar VPS pe chal raha hai, toh data/live_remote.json banao (git mein nahi jaata):
#     {"url": "http://<VPS IP>:8001", "token": "<VPS ka token>"}
# Tab ye endpoint data VPS ke live_api.py se laata hai. File na ho -> isi computer ka data/live.
LIVE = ROOT / "data" / "live"
REMOTE_FILE = ROOT / "data" / "live_remote.json"


def _remote_cfg() -> dict | None:
    if not REMOTE_FILE.exists():
        return None
    try:
        cfg = json.loads(REMOTE_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None
    return cfg if cfg.get("url") and cfg.get("token") else None


def _remote_get(remote: dict, path: str, params: dict, timeout: float = 10) -> dict:
    """VPS ke live_api se JSON lao. Gadbad ho toh {"status": "unreachable", ...}."""
    import urllib.error
    import urllib.parse
    import urllib.request
    url = remote["url"].rstrip("/") + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"X-Live-Token": remote["token"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()) | {"source": "vps"}
    except urllib.error.HTTPError as e:
        msg = "VPS rejected the token (check data/live_remote.json)" if e.code == 401 else f"VPS error {e.code}"
        return {"status": "unreachable", "source": "vps", "error": msg}
    except Exception as e:  # noqa: BLE001 - VPS band / IP badla / firewall
        return {"status": "unreachable", "source": "vps", "error": f"Cannot reach VPS: {e.__class__.__name__}"}


@app.get("/api/live/chart")
def live_chart(symbol: str = "EURUSD", tf: str = "M15"):
    """Live chart: bot ke MT5 candles + zones jo woh dekh raha hai + is pair ke trades."""
    from src.api import live_status as ls
    if not symbol.isalpha() or len(symbol) > 10:
        raise HTTPException(400, "bad symbol")
    remote = _remote_cfg()
    if remote is None:
        return ls.build_chart(LIVE, symbol.upper(), tf) | {"source": "local"}
    return _remote_get(remote, "/api/live/chart", {"symbol": symbol.upper(), "tf": tf})


@app.get("/api/live/ticks")
def live_ticks():
    """Halka live data (har 1 s): bid/ask, forming candle, khule trades -> live P&L + chalti candle."""
    from src.api import live_status as ls
    remote = _remote_cfg()
    if remote is None:
        return ls.build_ticks(LIVE) | {"source": "local"}
    return _remote_get(remote, "/api/live/ticks", {}, timeout=4)


@app.get("/api/live/orb")
def live_orb():
    """ORB shadow (src/live/orb.py): account, aaj ke dabbe, ORB trades, live stats (VPS ya isi PC ka data)."""
    from src.api import live_status as ls
    remote = _remote_cfg()
    if remote is None:
        return ls.build_orb(LIVE) | {"source": "local"}
    r = _remote_get(remote, "/api/live/orb", {})
    if r.get("error", "").startswith("VPS error 404"):
        r["error"] = "VPS live API is an old version: git pull + restart LIVE API on the VPS"
    return r


@app.get("/api/live/real")
def live_real():
    """REAL DEMO account panel (orders the bot sends to the MT5 demo): VPS or this PC."""
    from src.api import live_status as ls
    remote = _remote_cfg()
    if remote is None:
        return ls.build_real_account(LIVE) | {"source": "local"}
    r = _remote_get(remote, "/api/live/real", {})
    if r.get("error", "").startswith("VPS error 404"):
        r["error"] = "VPS live API is an old version: git pull + restart LIVE API on the VPS"
    return r


@app.get("/api/live/weekly")
def live_weekly():
    """Weekend report (bot writes it every Saturday NY): your AVOID / TAKE / EXIT / CLOSE decisions vs the bot."""
    from src.live import weekly_report
    remote = _remote_cfg()
    if remote is None:
        return (weekly_report.latest(LIVE) or {"status": "none"}) | {"source": "local"}
    r = _remote_get(remote, "/api/live/weekly", {})
    if r.get("error", "").startswith("VPS error 404"):
        r["error"] = "VPS live API is an old version: git pull + restart LIVE API on the VPS"
    return r


@app.get("/api/orb/backtest")
def orb_backtest():
    """ORB backtest numbers (walk-forward 2019-26) for the ORB stats page: data/research/orb_backtest_stats.json."""
    f = ROOT / "data" / "research" / "orb_backtest_stats.json"
    if not f.exists():
        raise HTTPException(404, "run: python src\\research\\orb_train.py")
    meta = ROOT / "models" / "orb_meta.json"
    out = json.loads(f.read_text())
    if meta.exists():
        m = json.loads(meta.read_text())
        out["model"] = {k: m.get(k) for k in ("trained_at", "rows", "data_to", "threshold", "keep", "tp_r", "tight_q30")}
    return out


@app.get("/api/orb/research/report")
def orb_research_report(pairs: str = "EURUSD,GBPUSD,USDJPY", ranges: str = "ASIA,LON30,LON60", comp: str = "nr4_or_tight",
                        ml: bool = True, tp: str = "2R", one_per_day: bool = True, direction: str = "all",
                        y0: int = 2016, y1: int = 2026):
    """Research -> ORB filters: PF / trades / equity for any combination (downloaded data, laptop only)."""
    from src.api import orb_research as orr
    try:
        return orr.report([p for p in pairs.split(",") if p], [r for r in ranges.split(",") if r], comp, ml, tp,
                          one_per_day, direction, y0, y1)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/orb/research/chart")
def orb_research_chart(symbol: str, start: str, end: str):
    """Research -> ORB history chart: ranges + trades of one pair in a date window."""
    from src.api import orb_research as orr
    if not symbol.isalpha() or len(symbol) > 10:
        raise HTTPException(400, "bad symbol")
    s, e = int(pd.Timestamp(start).timestamp()), int(pd.Timestamp(end).timestamp())
    try:
        return {"symbol": symbol.upper(), "ranges": orr.chart(symbol.upper(), s, e)}
    except FileNotFoundError as e_:
        raise HTTPException(404, str(e_)) from e_


@app.get("/api/orb/research/brain")
def orb_research_brain():
    """Research -> ORB ML brain: features, score buckets, threshold."""
    from src.api import orb_research as orr
    try:
        return orr.brain()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/live/control")
def live_control(action: str, symbol: str, zone: int | None = None, direction: str | None = None,
                 fill_time: int | None = None, exit: str | None = None):  # noqa: A002
    """Dashboard buttons: AVOID / TAKE / EXIT zone, EXIT / CLOSE zone trade, CLOSE ORB trade -> bot (VPS ya isi PC pe)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    from src.live import control
    remote = _remote_cfg()
    if remote is None:
        try:
            c = control.apply(LIVE, action, symbol, zone=zone, direction=direction, fill_time=fill_time, exit=exit)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True, **c, "source": "local"}
    params = {k: v for k, v in dict(action=action, symbol=symbol, zone=zone, direction=direction,
                                     fill_time=fill_time, exit=exit).items() if v is not None}
    url = remote["url"].rstrip("/") + "/api/live/control?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST", headers={"X-Live-Token": remote["token"]})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode()) | {"source": "vps"}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(502, "VPS live API is an old version: git pull + restart LIVE API on the VPS") from e
        raise HTTPException(502, "VPS rejected the token" if e.code == 401 else f"VPS error {e.code}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Cannot reach VPS: {e.__class__.__name__}") from e


@app.get("/api/live")
def live_status(decisions: int = 40, trades: int = 30):
    """Shadow bot ka live hisaab: account, khule trades, band trades, decisions, what-if summary."""
    from src.api import live_status as ls

    remote = _remote_cfg()
    if remote is None:
        return ls.build(LIVE, decisions, trades) | {"source": "local"}
    return _remote_get(remote, "/api/live", {"decisions": decisions, "trades": trades})


# ----------------------------------------------------------------------
# WebSocket: live events
# ----------------------------------------------------------------------
@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await bus.connect(ws)
    try:
        while True:
            await ws.receive_text()      # dashboard kuch bheje toh ignore; connection zinda rakho
    except WebSocketDisconnect:
        bus.disconnect(ws)
