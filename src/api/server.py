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
