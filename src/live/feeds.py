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
        self._check_offset()

    def _check_offset(self):
        """Abhi ka measured offset (tick time - UTC) vs 'NY+7' rule."""
        mt5 = self.mt5
        ticks = [mt5.symbol_info_tick(s) for s in ("EURUSD", "USDJPY", "GBPUSD")]
        ticks = [t for t in ticks if t is not None]
        if not ticks:
            return
        now = datetime.now(timezone.utc)
        server = max(t.time for t in ticks)
        measured = round((server - now.timestamp()) / 3600)
        stale = now.timestamp() - (server - measured * 3600) > 600
        ny_rule = int((pd.Timestamp(now).tz_convert(NY).utcoffset().total_seconds() / 3600) + 7)
        if stale:
            self.log(f"feed: market band hai (purana tick); NY+7 rule use (abhi UTC{ny_rule:+d})")
        elif measured != ny_rule:
            self.mode, self.fixed_hours = "fixed", measured
            self.log(f"feed WARNING: broker time UTC{measured:+d} != NY+7 rule (UTC{ny_rule:+d}). Fixed offset use kar raha hoon.")
        else:
            self.log(f"feed: broker server time = NY+7 (abhi UTC{measured:+d}) - OK")

    def _to_utc(self, server_secs: np.ndarray) -> pd.DatetimeIndex:
        naive = pd.to_datetime(server_secs, unit="s")            # server wall-clock time (bina timezone)
        if self.mode == "fixed":
            return (naive - pd.Timedelta(hours=self.fixed_hours)).tz_localize("UTC")
        ny_wall = naive - pd.Timedelta(hours=7)
        return ny_wall.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")

    def bars(self, symbol: str):
        mt5 = self.mt5
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

    def bars(self, symbol: str):
        m5 = self.data[symbol]
        t = self.clock
        return m5[(m5.index < t) & (m5.index >= t - self.hist)], t
