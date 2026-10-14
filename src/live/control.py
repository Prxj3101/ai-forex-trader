r"""
MANUAL CONTROL: dashboard se tere haath ke buttons -> bot tak.

  AVOID     = "is zone pe trade mat lena" (watchlist ka zone). UNAVOID = wapas allow.
  TAKE      = "ML LOW / value area / quiet ki wajah se ruka hai, phir bhi le lo" (15 Oct). UNTAKE = wapas.
              Risk rules (news, kill switch, daily limit, max open, 1 per pair, lot) tab bhi lagte hain.
  EXIT      = zone ka exit trade se PEHLE badlo: 1:1 (tp1) ya TRAIL (exit brain ki jagah). UNEXIT = brain wapas.
  TRADE_EXIT= khule zone trade ka exit BEECH mein badlo (tp1 <-> trail). Trail -> 1:1 aur trade +1R se upar:
              turant market pe band (1R pe pehle hi TP hota).
  CLOSE     = khula zone shadow trade abhi market price pe band.
  ORB_CLOSE = khula ORB shadow trade abhi market price pe band (15 Oct).

Kaise: dashboard -> laptop backend -> VPS live_api (token ke saath) -> data/live/control.json likhta hai.
Bot (shadow.py / orb.py) har ~1 sec is file ko dekhta hai aur command lagata hai.
Har manual kaam data/live/user_actions.csv mein bhi jaata hai (bot likhta hai) -> weekend report + mahine ki training.

control.json: {"avoid": [{symbol, zone, direction, at}], "take": [...same], "exit": [{symbol, zone, direction, exit, at}],
               "trade_exit": [{id, symbol, fill_time, exit, at}], "close": [{id, symbol, fill_time, at}],
               "orb_close": [{id, symbol, fill_time, at}]}
  zone / fill_time = unix seconds (UTC). Ho chuke command ids state files mein (close_done).
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

FILE = "control.json"
_lock = threading.Lock()
ZONE_KEEP_S = 10 * 86400           # zone 5 trading din tak zinda rehta hai -> 10 din baad list se hata do
CMD_KEEP_S = 86400                 # close / trade_exit command 1 din baad purana
ZONE_LISTS = ("avoid", "take", "exit")
CMD_LISTS = ("close", "trade_exit", "orb_close")
EXITS = ("tp1", "trail")


def load(live: Path) -> dict:
    p = live / FILE
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001 - adhuri likhi file -> agli baar
        d = {}
    return {k: list(d.get(k) or []) for k in ZONE_LISTS + CMD_LISTS}


def avoid_key(symbol: str, zone: int, direction: str) -> tuple:
    return (str(symbol), int(zone), str(direction))


zone_key = avoid_key


def _check(symbol, direction=None):
    if not isinstance(symbol, str) or not symbol.isalpha() or len(symbol) > 10:
        raise ValueError("bad symbol")
    if direction is not None and direction not in ("bull", "bear"):
        raise ValueError("bad direction")


def apply(live: Path, action: str, symbol: str, zone: int | None = None, direction: str | None = None,
          fill_time: int | None = None, exit: str | None = None) -> dict:  # noqa: A002 - query param name
    """Ek command control.json mein likho (atomic). ValueError = galat input."""
    symbol = (symbol or "").upper()
    now = int(time.time())
    with _lock:
        c = load(live)
        for k in ZONE_LISTS:
            c[k] = [a for a in c[k] if now - int(a.get("at", now)) < ZONE_KEEP_S]
        for k in CMD_LISTS:
            c[k] = [x for x in c[k] if now - int(x.get("at", now)) < CMD_KEEP_S]
        base = action[2:] if action.startswith("un") else action
        if base in ZONE_LISTS:
            _check(symbol, direction)
            if zone is None:
                raise ValueError("zone missing")
            key = zone_key(symbol, zone, direction)
            if base == "exit" and not action.startswith("un") and exit not in EXITS:
                raise ValueError("exit must be tp1 or trail")
            c[base] = [a for a in c[base] if zone_key(a["symbol"], a["zone"], a["direction"]) != key]
            if not action.startswith("un"):
                row = {"symbol": symbol, "zone": int(zone), "direction": direction, "at": now}
                if base == "exit":
                    row["exit"] = exit
                c[base].append(row)
            if base == "avoid" and not action.startswith("un"):      # avoid aur take ek saath nahi
                c["take"] = [a for a in c["take"] if zone_key(a["symbol"], a["zone"], a["direction"]) != key]
            if base == "take" and not action.startswith("un"):
                c["avoid"] = [a for a in c["avoid"] if zone_key(a["symbol"], a["zone"], a["direction"]) != key]
        elif action in CMD_LISTS:
            _check(symbol)
            if fill_time is None:
                raise ValueError("fill_time missing")
            if action == "trade_exit" and exit not in EXITS:
                raise ValueError("exit must be tp1 or trail")
            same = [x for x in c[action] if x["symbol"] == symbol and int(x["fill_time"]) == int(fill_time)]
            if action == "trade_exit":
                c[action] = [x for x in c[action] if x not in same]      # naya exit purane command ki jagah
                same = []
            if not same:
                row = {"id": secrets.token_hex(6), "symbol": symbol, "fill_time": int(fill_time), "at": now}
                if action == "trade_exit":
                    row["exit"] = exit
                c[action].append(row)
        else:
            raise ValueError("unknown action")
        live.mkdir(parents=True, exist_ok=True)
        tmp = live / (FILE + ".tmp")
        tmp.write_text(json.dumps(c))
        tmp.replace(live / FILE)
        return c


def log_action(live: Path, **row):
    """data/live/user_actions.csv: har manual kaam (bot likhta hai jab command lagta hai)."""
    import pandas as pd
    cols = ["at", "kind", "symbol", "direction", "zone", "fill_time", "from_", "to", "r_now", "score", "note"]
    row.setdefault("at", pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    f = live / "user_actions.csv"
    try:
        pd.DataFrame([row]).reindex(columns=cols).to_csv(f, mode="a", header=not f.exists(), index=False)
    except Exception:  # noqa: BLE001 - log kabhi bot na roke
        pass
