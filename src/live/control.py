r"""
MANUAL CONTROL: dashboard se tere haath ke 2 button -> bot tak.

  AVOID  = "is zone pe trade mat lena" (watchlist ka zone). UNDO = wapas allow.
  CLOSE  = "ye khula shadow trade abhi market price pe band karo".

Kaise: dashboard -> laptop backend -> VPS live_api (token ke saath) -> data/live/control.json likhta hai.
Bot (shadow.py) har ~1 sec is file ko dekhta hai aur command lagata hai.
Sirf yahi 2 kaam ho sakte hain - na settings badal sakte, na bot rok sakte, na naya trade khol sakte.

control.json: {"avoid": [{symbol, zone, direction, at}], "close": [{id, symbol, fill_time, at}]}
  zone / fill_time = unix seconds (UTC). Bot jo close kar chuka, uske id shadow_state.json mein (close_done).
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

FILE = "control.json"
_lock = threading.Lock()
AVOID_KEEP_S = 10 * 86400          # zone 5 trading din tak zinda rehta hai -> 10 din baad list se hata do
CLOSE_KEEP_S = 86400               # close command 1 din baad purana


def load(live: Path) -> dict:
    p = live / FILE
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001 - adhuri likhi file -> agli baar
        d = {}
    return {"avoid": list(d.get("avoid") or []), "close": list(d.get("close") or [])}


def avoid_key(symbol: str, zone: int, direction: str) -> tuple:
    return (str(symbol), int(zone), str(direction))


def _check(symbol, direction=None):
    if not isinstance(symbol, str) or not symbol.isalpha() or len(symbol) > 10:
        raise ValueError("bad symbol")
    if direction is not None and direction not in ("bull", "bear"):
        raise ValueError("bad direction")


def apply(live: Path, action: str, symbol: str, zone: int | None = None, direction: str | None = None,
          fill_time: int | None = None) -> dict:
    """Ek command control.json mein likho (atomic). ValueError = galat input."""
    symbol = (symbol or "").upper()
    now = int(time.time())
    with _lock:
        c = load(live)
        c["avoid"] = [a for a in c["avoid"] if now - int(a.get("at", now)) < AVOID_KEEP_S]
        c["close"] = [x for x in c["close"] if now - int(x.get("at", now)) < CLOSE_KEEP_S]
        if action in ("avoid", "unavoid"):
            _check(symbol, direction)
            if zone is None:
                raise ValueError("zone missing")
            key = avoid_key(symbol, zone, direction)
            c["avoid"] = [a for a in c["avoid"] if avoid_key(a["symbol"], a["zone"], a["direction"]) != key]
            if action == "avoid":
                c["avoid"].append({"symbol": symbol, "zone": int(zone), "direction": direction, "at": now})
        elif action == "close":
            _check(symbol)
            if fill_time is None:
                raise ValueError("fill_time missing")
            if not any(x["symbol"] == symbol and int(x["fill_time"]) == int(fill_time) for x in c["close"]):
                c["close"].append({"id": secrets.token_hex(6), "symbol": symbol, "fill_time": int(fill_time), "at": now})
        else:
            raise ValueError("unknown action")
        live.mkdir(parents=True, exist_ok=True)
        tmp = live / (FILE + ".tmp")
        tmp.write_text(json.dumps(c))
        tmp.replace(live / FILE)
        return c
