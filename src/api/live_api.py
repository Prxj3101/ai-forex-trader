r"""
LIVE API (VPS ke liye): chhota server jo shadow bot ka live hisaab dikhata hai.

Kyun: bot VPS pe chalta hai, dashboard laptop pe. Laptop ka backend yahan se data mangta hai.
Likh sirf EK cheez sakta hai: data/live/control.json (dashboard ke AVOID / UNDO / CLOSE buttons).
Iske alawa kuch nahi badal sakta (na bot rok sakta, na naya trade, na settings, na retrain).

Suraksha:
  - har request mein token chahiye (header "X-Live-Token"). Token data/live_api_token.txt mein
    rehta hai (data/ git mein nahi jaata). Pehli baar chalane pe khud banta hai aur screen pe dikhta hai.
  - AWS security group + Windows Firewall mein port sirf tere ghar ke IP ke liye khulega.

Chalana (VPS pe, venv active):
    python src\api\live_api.py            (port 8001)
    python src\api\live_api.py --show-token
Ya start_live_api.bat (Startup folder mein shortcut -> VPS restart ke baad khud chalu).

Endpoints:
    GET /health    -> {"ok": true}   (token nahi chahiye, sirf "server zinda hai?")
    GET /api/live  -> wahi JSON jo dashboard ka LIVE panel use karta hai (token chahiye)
    GET /api/live/chart?symbol=EURUSD&tf=M15 -> live MT5 candles + zones + trades (token chahiye)
    GET /api/live/ticks -> bid/ask + forming candle + khule trades, har ~1 s (token chahiye)
    POST /api/live/control?action=avoid|unavoid|close&symbol=..&zone=..&direction=..&fill_time=.. (token chahiye)
"""
from __future__ import annotations

import argparse
import hmac
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Header, HTTPException  # noqa: E402

from src.api import live_status  # noqa: E402

LIVE = ROOT / "data" / "live"
TOKEN_FILE = ROOT / "data" / "live_api_token.txt"
PORT = 8001


def get_token() -> str:
    """Token file se padho; na ho toh naya random token banao aur save karo."""
    if TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text().strip()
        if tok:
            return tok
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(tok)
    print(f"live api: naya token banaya -> {TOKEN_FILE}")
    return tok


TOKEN = get_token()
app = FastAPI(title="Forex bot live (read-only)", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/live")
def live(decisions: int = 40, trades: int = 30, x_live_token: str = Header(default="")):
    if not hmac.compare_digest(x_live_token, TOKEN):
        raise HTTPException(401, "bad token")
    decisions, trades = min(max(decisions, 1), 200), min(max(trades, 1), 200)
    return live_status.build(LIVE, decisions, trades)


@app.get("/api/live/chart")
def live_chart(symbol: str = "EURUSD", tf: str = "M15", x_live_token: str = Header(default="")):
    if not hmac.compare_digest(x_live_token, TOKEN):
        raise HTTPException(401, "bad token")
    if not symbol.isalpha() or len(symbol) > 10:          # sirf "EURUSD" jaisa naam (file path safe)
        raise HTTPException(400, "bad symbol")
    return live_status.build_chart(LIVE, symbol.upper(), tf)


@app.get("/api/live/ticks")
def live_ticks(x_live_token: str = Header(default="")):
    """Halka: bid/ask + forming candle + khule trades (dashboard har 1 s, live P&L)."""
    if not hmac.compare_digest(x_live_token, TOKEN):
        raise HTTPException(401, "bad token")
    return live_status.build_ticks(LIVE)


@app.post("/api/live/control")
def live_control(action: str, symbol: str, zone: int | None = None, direction: str | None = None,
                 fill_time: int | None = None, x_live_token: str = Header(default="")):
    """Sirf 2 manual buttons: AVOID/UNDO zone, CLOSE khula shadow trade. Bot ~1 s mein lagata hai."""
    if not hmac.compare_digest(x_live_token, TOKEN):
        raise HTTPException(401, "bad token")
    from src.live import control
    try:
        c = control.apply(LIVE, action, symbol, zone=zone, direction=direction, fill_time=fill_time)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "avoid": c["avoid"], "close": c["close"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--show-token", action="store_true", help="token dikhao aur band ho jao")
    a = ap.parse_args()
    if a.show_token:
        print(TOKEN)
        raise SystemExit
    import uvicorn
    print(f"live api: port {a.port}, data folder {LIVE} (read + manual AVOID/CLOSE buttons only)")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
