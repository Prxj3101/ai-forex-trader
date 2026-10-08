r"""
MONITOR: bot ki "sehat" dashboard tak pahunchane ke liye 2 cheezein (sirf likhta hai, trading pe koi asar nahi).

1. Tee      -> jo bhi bot window mein print hota hai (aur error / traceback), wahi data/live/bot.log mein bhi.
               File 2 MB se badi ho toh bot.log.1 bana ke nayi shuru (disk nahi bharti).
2. write_health() -> har ~10 sec data/live/health.json:
               MT5 connected?, ping, demo account balance, teeno pairs ke live bid/ask/spread,
               market khula hai ya band, VPS CPU / RAM / disk, bot + MT5 ki RAM, kab start hua, kitni baar restart.

Dashboard (src/api/live_status.py) ye files padh ke dikhata hai, taaki RDP kholne ki zaroorat na pade.
Koi bhi gadbad ho toh yahan ka error bot ko kabhi rokta nahi (sab try/except mein).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG_MAX_BYTES = 2_000_000
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}


class Tee:
    """sys.stdout / sys.stderr ki jagah: screen pe bhi likho aur bot.log mein bhi."""

    def __init__(self, stream, path: Path):
        self.stream, self.path = stream, path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8", errors="replace", buffering=1)

    def _rotate(self):
        try:
            if self.f.tell() > LOG_MAX_BYTES:
                self.f.close()
                old = self.path.with_suffix(".log.1")
                old.unlink(missing_ok=True)
                self.path.replace(old)
                self.f = open(self.path, "a", encoding="utf-8", errors="replace", buffering=1)
        except Exception:  # noqa: BLE001 - rotate na ho sake toh bas likhte raho
            try:
                if self.f.closed:
                    self.f = open(self.path, "a", encoding="utf-8", errors="replace", buffering=1)
            except Exception:  # noqa: BLE001
                pass

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.f.write(s)
            if "\n" in s:
                self._rotate()
        except Exception:  # noqa: BLE001
            pass
        return len(s)

    def flush(self):
        for x in (self.stream, self.f):
            try:
                x.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):          # encoding waghera original stream se
        return getattr(self.stream, name)


def start_log(live_dir: Path):
    """Bot window ka sab kuch data/live/bot.log mein bhi likho."""
    path = live_dir / "bot.log"
    if not isinstance(sys.stdout, Tee):
        sys.stdout = Tee(sys.stdout, path)
        sys.stderr = Tee(sys.stderr, path)
    print(f"\n===== bot start {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S} UTC (pid {os.getpid()}) =====")


def _restarts(live_dir: Path) -> dict:
    f = live_dir / "restarts.log"
    if not f.exists():
        return {"starts": 0, "last": None}
    try:
        lines = [x.strip() for x in f.read_text(errors="replace").splitlines() if "bot start" in x]
        return {"starts": len(lines), "last": lines[-1] if lines else None}
    except Exception:  # noqa: BLE001
        return {"starts": None, "last": None}


def _system(mt5_pid_cache: dict) -> dict:
    try:
        import psutil
    except ImportError:
        return {}
    out = {}
    try:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage(os.path.abspath(os.sep))
        out.update(cpu_pct=psutil.cpu_percent(interval=None), ram_used_mb=round((vm.total - vm.available) / 2**20),
                   ram_total_mb=round(vm.total / 2**20), disk_free_gb=round(du.free / 2**30, 1),
                   disk_total_gb=round(du.total / 2**30, 1), boot_time=float(psutil.boot_time()))
        out["bot_ram_mb"] = round(psutil.Process().memory_info().rss / 2**20)
        # MT5 terminal process (naam terminal64.exe); pid yaad rakho taaki har baar sab process na dhoondhne pade
        p = None
        pid = mt5_pid_cache.get("pid")
        if pid and psutil.pid_exists(pid):
            p = psutil.Process(pid)
        else:
            for q in psutil.process_iter(["name"]):
                if (q.info.get("name") or "").lower().startswith("terminal64"):
                    p = q
                    mt5_pid_cache["pid"] = q.pid
                    break
        out["mt5_ram_mb"] = round(p.memory_info().rss / 2**20) if p else None
    except Exception:  # noqa: BLE001
        pass
    return out


_MT5_PID: dict = {}


def _prices_forming(feed, eng, now: pd.Timestamp):
    """Har pair ka abhi ka bid/ask + chal rahi (forming) M5 candle. (prices, forming, market_open)"""
    mt5 = feed.mt5
    prices, forming, fresh = {}, {}, False
    for s in eng.symbols:
        t = mt5.symbol_info_tick(s)
        if t is None:
            continue
        tick_utc = feed._to_utc(np.array([t.time]))[0]
        age = (now - tick_utc).total_seconds() if pd.notna(tick_utc) else None
        fresh = fresh or (age is not None and age < 600)
        pip = PIP.get(s, 0.0001)
        prices[s] = {"bid": float(t.bid), "ask": float(t.ask),
                     "spread_pips": round((t.ask - t.bid) / pip, 2),
                     "tick_age_sec": round(age) if age is not None else None,
                     "time_ms": (int(feed._to_utc(np.array([t.time_msc // 1000]))[0].value // 1_000_000)
                                 + int(t.time_msc % 1000)) if pd.notna(tick_utc) else None}
    # abhi chal rahi (forming) M5 candle -> dashboard chart TradingView jaisa hilta rahe
    for s in eng.symbols:
        try:
            r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 1)
            if r is None or len(r) == 0:
                continue
            t0 = feed._to_utc(np.array([r[0]["time"]]))[0]
            if pd.notna(t0):
                forming[s] = [int(t0.timestamp()), float(r[0]["open"]), float(r[0]["high"]),
                              float(r[0]["low"]), float(r[0]["close"])]
        except Exception:  # noqa: BLE001
            continue
    return prices, forming, fresh


def write_ticks(feed, eng, live_dir: Path):
    """data/live/ticks.json: sirf bid/ask + forming candle (har ~1 sec). Halka file -> dashboard live P&L + candle."""
    try:
        mt5 = getattr(feed, "mt5", None)
        if mt5 is None:
            return None
        now = pd.Timestamp.now(tz="UTC")
        prices, forming, _ = _prices_forming(feed, eng, now)
        tmp = live_dir / "ticks.json.tmp"
        tmp.write_text(json.dumps({"updated": now.isoformat(), "prices": prices, "forming": forming}))
        tmp.replace(live_dir / "ticks.json")
        return prices                     # shadow loop isse armed limits ka turant fill check karta hai
    except Exception as e:  # noqa: BLE001 - kabhi bot ko na roke
        if not getattr(write_ticks, "_warned", False):
            print(f"monitor: ticks likhne mein dikkat ({e.__class__.__name__}: {e}) - bot chalta rahega")
            write_ticks._warned = True
        return None


def write_health(feed, eng, live_dir: Path, started_at: pd.Timestamp, guard=None):
    """data/live/health.json likho. Bot ke main loop se har ~10 sec bulaya jaata hai."""
    try:
        now = pd.Timestamp.now(tz="UTC")
        h: dict = {"updated": now.isoformat(), "started_at": started_at.isoformat(), "pid": os.getpid()}
        mt5 = getattr(feed, "mt5", None)
        prices, fresh = {}, False
        if mt5 is not None:
            ti, ai = mt5.terminal_info(), mt5.account_info()
            h["mt5"] = {
                "connected": bool(ti.connected) if ti else False,
                "trade_allowed": bool(ti.trade_allowed) if ti else False,
                "ping_ms": round(ti.ping_last / 1000, 1) if ti and ti.ping_last else None,
                "build": ti.build if ti else None,
                "login": ai.login if ai else None, "server": ai.server if ai else None,
                "balance": float(ai.balance) if ai else None, "equity": float(ai.equity) if ai else None,
                "currency": ai.currency if ai else None,
                "broker_time": ("NY+7" if feed.mode == "ny+7" else f"UTC{feed.fixed_hours:+d}"),
                "broker_time_checked": not getattr(feed, "unverified", False),
            }
            prices, forming, fresh = _prices_forming(feed, eng, now)
            h["forming"] = forming
        h["prices"] = prices
        h["market_open"] = fresh
        h["system"] = _system(_MT5_PID)
        h["restarts"] = _restarts(live_dir)
        if guard is not None:
            try:
                h["news_blocked"] = {s: (guard.blocked(s, now) or None) for s in eng.symbols}
            except Exception:  # noqa: BLE001
                pass
            try:   # is hafte ki high-impact news (dashboard news panel)
                from src import config
                from src.features.news import PAIR_CCY
                ev = guard.events
                h["news"] = {
                    "block_before_min": config.NEWS_BLOCK_BEFORE_MIN if config.NEWS_BLOCK else None,
                    "block_after_min": config.NEWS_BLOCK_AFTER_MIN if config.NEWS_BLOCK else None,
                    "events": [
                        {"time": int(pd.Timestamp(r.time).timestamp()), "currency": str(r.currency), "event": str(r.event),
                         "pairs": [s for s in eng.symbols if str(r.currency) in PAIR_CCY.get(s, ())]}
                        for r in ev.sort_values("time").itertuples(index=False)
                    ] if ev is not None and len(ev) else [],
                }
            except Exception:  # noqa: BLE001
                pass
        tmp = live_dir / "health.json.tmp"
        tmp.write_text(json.dumps(h, default=str))
        tmp.replace(live_dir / "health.json")
    except Exception as e:  # noqa: BLE001 - health kabhi bot ko na roke
        if not getattr(write_health, "_warned", False):
            print(f"monitor: health likhne mein dikkat ({e.__class__.__name__}: {e}) - bot chalta rahega")
            write_health._warned = True


def tail_lines(path: Path, n: int = 150, max_bytes: int = 120_000) -> list[str]:
    """Badi log file ki sirf aakhri n lines."""
    if not path.exists():
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read().decode("utf-8", "replace")
        lines = chunk.splitlines()
        if size > max_bytes and lines:
            lines = lines[1:]               # pehli line aadhi kati ho sakti hai
        return [x.rstrip() for x in lines if x.strip()][-n:]
    except Exception:  # noqa: BLE001
        return []

