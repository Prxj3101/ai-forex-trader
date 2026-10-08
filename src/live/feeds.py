r"""
DATA FEEDS for the live engine. Dono ek hi cheez dete hain:
    bars(symbol) -> (closed_m5, bar_time)
        closed_m5 : BAND ho chuki M5 candles (UTC index; open/high/low/close/volume/spread; BID prices)
        bar_time  : abhi khuli M5 candle ka open time (UTC)

MT5Feed    : asli MT5 terminal (demo account) se
ReplayFeed : purane Dukascopy data se, candle-by-candle (engine ko bina market ke test karne ke liye)

BROKER TIME: MT5 candles broker ke SERVER time mein aate hain, UTC mein nahi.
Zyada tar forex brokers: server time = New York time + 7h (unka din NY 5pm pe khatam). Isse purane candles
ka DST bhi sahi hota hai. Start pe check karte hain ki abhi ka measured offset is rule se match karta hai;
match na ho toh warning + measured fixed offset use (config ya broker badalne pe dekhna).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
NY = "America/New_York"


def _fresh_offset(server: float, now: float):
    """Tick taaza (pichhle 10 min) ho toh broker offset (ghante, -12..+14), warna None."""
    h = round((server - now) / 3600)
    if -12 <= h <= 14 and -600 < (server - h * 3600) - now < 120:
        return int(h)
    return None


class MT5Feed:
    def __init__(self, history_days: int = 200, log=print):
        import MetaTrader5 as mt5
        self.mt5 = mt5
        if not mt5.initialize():
            raise SystemExit(f"MT5 se connect nahi hua: {mt5.last_error()} (MT5 desktop khula + login hai?)")
        acc = mt5.account_info()
        if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            mt5.shutdown()
            raise SystemExit("SAFETY: MT5 mein DEMO account login nahi hai. Bot sirf demo pe chalega.")
        self.n_bars = history_days * 288
        self.log = log
        self.mode = "ny+7"
        self.fixed_hours = None
        self.unverified = False          # True = offset abhi pakka nahi, market khulte hi dobara naapo
        self._check_offset()

    def _set(self, hours, ny_rule, how):
        """hours = broker server time - UTC (ghante)."""
        if hours == ny_rule:
            self.mode, self.fixed_hours = "ny+7", None
            self.log(f"feed: broker server time = NY+7 (abhi UTC{hours:+d}) [{how}]")
        else:
            self.mode, self.fixed_hours = "fixed", int(hours)
            self.log(f"feed: broker server time = UTC{hours:+d} (fixed) [{how}]")

    def _check_offset(self):
        """Broker ki ghadi ka UTC offset tay karo: config > live tick > weekend close."""
        from src import config
        mt5 = self.mt5
        now = datetime.now(timezone.utc)
        now_ny = pd.Timestamp(now).tz_convert(NY)
        ny_rule = int(now_ny.utcoffset().total_seconds() / 3600 + 7)
        self.unverified = False

        cfg = getattr(config, "BROKER_UTC_OFFSET", None)
        if cfg is not None:
            if cfg == "ny+7":
                self.mode, self.fixed_hours = "ny+7", None
                self.log("feed: broker server time = NY+7 [config]")
            else:
                self._set(int(cfg), ny_rule, "config")
            return

        ticks = [mt5.symbol_info_tick(s) for s in ("EURUSD", "USDJPY", "GBPUSD")]
        ticks = [t for t in ticks if t is not None]
        if not ticks:
            self.unverified = True
            self.log("feed WARNING: koi tick nahi mila; NY+7 maan raha hoon, market khulte hi dobara check")
            return
        server = max(t.time for t in ticks)                       # server wall-clock, epoch jaisa number
        measured = _fresh_offset(server, now.timestamp())
        if measured is not None:
            self._set(measured, ny_rule, "live tick")
            return

        # Market band: aakhri tick = weekly close (Friday 5 PM New York) ke just pehle
        days_back = (now_ny.weekday() - 4) % 7                    # Friday = 4
        close_ny = (now_ny - pd.Timedelta(days=days_back)).normalize() + pd.Timedelta(hours=17)
        if close_ny > now_ny:
            close_ny -= pd.Timedelta(days=7)
        weekend = now_ny - close_ny < pd.Timedelta(hours=49)     # Fri 5pm -> Sun 6pm NY
        diff_h = (server - close_ny.tz_convert("UTC").timestamp()) / 3600
        inferred = round(diff_h)
        if weekend and abs(diff_h - inferred) < 0.25 and -12 <= inferred <= 14:
            close_offset = int(close_ny.utcoffset().total_seconds() / 3600 + 7)
            # "NY+7" broker ka offset close ke waqt close_offset hota; abhi ke DST ke hisaab se ny_rule
            self._set(ny_rule if inferred == close_offset else inferred, ny_rule, "weekend close")
            self.unverified = True                                # market khulne pe live tick se confirm
        else:
            self.unverified = True
            self.log(f"feed WARNING: market band, offset pakka nahi (diff {diff_h:.2f}h); NY+7 maan raha hoon, "
                     f"market khulte hi dobara check. Pakka karna ho toh config.BROKER_UTC_OFFSET set karo.")

    def _recheck_if_needed(self):
        if not self.unverified:
            return
        mt5 = self.mt5
        ticks = [mt5.symbol_info_tick(s) for s in ("EURUSD", "USDJPY", "GBPUSD")]
        ticks = [t for t in ticks if t is not None]
        if not ticks:
            return
        server = max(t.time for t in ticks)
        if _fresh_offset(server, datetime.now(timezone.utc).timestamp()) is not None:   # tick taaza hai
            old = (self.mode, self.fixed_hours)
            self._check_offset()
            if (self.mode, self.fixed_hours) != old:
                self.log("feed: broker time badla (market khulne pe live check) - ab sahi offset use ho raha hai")

    def _to_utc(self, server_secs: np.ndarray) -> pd.DatetimeIndex:
        naive = pd.to_datetime(server_secs, unit="s")            # server wall-clock time (bina timezone)
        if self.mode == "fixed":
            return (naive - pd.Timedelta(hours=self.fixed_hours)).tz_localize("UTC")
        ny_wall = naive - pd.Timedelta(hours=7)
        return ny_wall.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")

    def bars(self, symbol: str):
        mt5 = self.mt5
        self._recheck_if_needed()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"{symbol} MT5 pe nahi mila")
        mt5.symbol_select(symbol, True)
        r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, self.n_bars)
        if r is None or len(r) < 1000:
            raise RuntimeError(f"{symbol}: MT5 se candles nahi aaye ({mt5.last_error()}). Tools>Options>Charts>Max bars badhao.")
        df = pd.DataFrame(r)
        idx = self._to_utc(df["time"].to_numpy())
        m5 = pd.DataFrame({"open": df["open"].to_numpy(), "high": df["high"].to_numpy(), "low": df["low"].to_numpy(),
                           "close": df["close"].to_numpy(), "volume": df["tick_volume"].to_numpy(float),
                           "spread": df["spread"].to_numpy(float) * info.point}, index=idx)
        m5 = m5[m5.index.notna()]
        m5.index.name = "time"
        bar_time = m5.index[-1]                     # aakhri = abhi khuli (forming) candle
        return m5.iloc[:-1], bar_time

    def _offset_secs(self, utc_secs: float) -> int:
        """broker server time - UTC (seconds) at a given UTC moment."""
        if self.mode == "fixed":
            return int(self.fixed_hours * 3600)
        ny = pd.Timestamp(utc_secs, unit="s", tz="UTC").tz_convert(NY)
        return int(ny.utcoffset().total_seconds()) + 7 * 3600

    def ticks(self, symbol: str, from_ms: int, to_ms: int):
        """Real MT5 ticks (bid/ask) with from_ms <= UTC time < to_ms. Returns (t_ms, ask, bid) numpy arrays
        (may be empty) or None if MT5 gave nothing usable. Used for tick-exact fills / SL / TP / trailing."""
        mt5 = self.mt5
        # MT5 dates = broker server wall-clock epoch (same as the times it returns). Small pad, then exact filter.
        pad = 120
        a = int(from_ms // 1000) + self._offset_secs(from_ms / 1000) - pad
        b = int(to_ms // 1000) + self._offset_secs(to_ms / 1000) + pad
        r = mt5.copy_ticks_range(symbol, a, b, mt5.COPY_TICKS_ALL)
        if r is not None and len(r) == 0:
            last = mt5.symbol_info_tick(symbol)
            if last is not None and last.time_msc >= from_ms + self._offset_secs(from_ms / 1000) * 1000:  # ticks exist, none came back
                if not getattr(self, "_tick_warned", False):
                    self.log("feed WARNING: copy_ticks_range gave nothing for a window that has ticks; retrying wider")
                    self._tick_warned = True
                r = mt5.copy_ticks_range(symbol, a - 4 * 3600, b + 4 * 3600, mt5.COPY_TICKS_ALL)
        if r is None:
            return None
        if len(r) == 0:
            return np.array([], dtype=np.int64), np.array([]), np.array([])
        msc = np.asarray(r["time_msc"], dtype=np.int64)
        utc = self._to_utc(msc // 1000)
        t_ms = utc.as_unit("ns").asi8 // 1_000_000 + (msc % 1000)
        ask, bid = np.asarray(r["ask"], float), np.asarray(r["bid"], float)
        ok = (t_ms >= from_ms) & (t_ms < to_ms) & (ask > 0) & (bid > 0) & ~pd.isna(utc)
        o = np.argsort(t_ms[ok], kind="stable")
        return t_ms[ok][o], ask[ok][o], bid[ok][o]

    def current_bar_time(self, symbol: str) -> pd.Timestamp:
        r = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M5, 0, 1)
        return self._to_utc(np.array([r[0]["time"]]))[0]

    def account(self) -> dict:
        a = self.mt5.account_info()
        return dict(login=a.login, server=a.server, balance=a.balance, equity=a.equity, currency=a.currency)

    def close(self):
        self.mt5.shutdown()


class ReplayFeed:
    """Purana data candle-by-candle. clock = abhi khuli candle ka time."""

    def __init__(self, symbols, start: str, end: str, history_days: int = 200):
        self.data = {s: pd.read_parquet(DATA / f"{s}_M5.parquet",
                                        columns=["open", "high", "low", "close", "volume", "spread"]) for s in symbols}
        self.hist = pd.Timedelta(days=history_days)
        ref = self.data[symbols[0]].index
        self.times = ref[(ref >= pd.Timestamp(start, tz="UTC")) & (ref < pd.Timestamp(end, tz="UTC"))]
        self.clock = None

    def __iter__(self):
        for t in self.times:
            self.clock = t
            yield t

    def ticks(self, symbol: str, from_ms: int, to_ms: int):
        """Replay: real Dukascopy ticks from data/ticks/store (None if that month is not converted)."""
        from src.backtest import tick_engine as te
        return te.ticks(symbol, int(from_ms), int(min(to_ms, self.clock.value // 1_000_000)))

    def bars(self, symbol: str):
        m5 = self.data[symbol]
        t = self.clock
        return m5[(m5.index < t) & (m5.index >= t - self.hist)], t
