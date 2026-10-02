r"""
Risk Agent: PROP FIRM NEWS RULES (account profile ke hisaab se).

Kya karta hai (profile "prop_master" mein, config.py dekho):
  1. Har news ke around ek "restricted window" banata hai:
        news   : [time - before, time + after]
        speech : [start - speech_before, start + duration + speech_after]
     Sirf us pair ke currencies ki news (EURUSD -> EUR + USD news).
  2. can_open(): window ke andar (aur window se thoda pehle) NAYA trade / pending limit order nahi.
  3. close_deadline(): jo trade event se 5 ghante se kam pehle khula, usko window shuru hone se
     pehle band karna hai -> deadline time deta hai. 5+ ghante purana trade exempt hai.
  4. apply_backtest(): yahi rules purane trades pe lagata hai (portfolio_sim ke liye):
        - fill window mein hua -> trade nahi hota (order cancel tha)
        - deadline se pehle exit nahi hua -> deadline pe M5 close pe band (reason "news")
        - pre_news="hold": trade chalta hai, par window mein profit se exit hua -> profit 0

Same code live bot mein bhi use hoga (Execution Agent har minute can_open / close_deadline puchega).
Profile mein news_rules=False (evaluation / real broker) -> sab allowed, kuch nahi badalta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.features import news  # noqa: E402

DATA = ROOT / "data" / "processed"


class NewsGuard:
    def __init__(self, events: pd.DataFrame, profile: dict | None = None, block: bool | None = None):
        self.p = profile or config.profile()
        self.on = bool(self.p.get("news_rules"))          # prop firm compliance (close before news etc.)
        self.block_on = config.NEWS_BLOCK if block is None else block   # NEWS BLOCK: no new trades near news
        self.events = events
        self._cache: dict[str, pd.DataFrame] = {}
        self._bcache: dict[str, pd.DataFrame] = {}

    def block_windows(self, symbol: str) -> pd.DataFrame:
        """NEWS BLOCK windows [event_time - before, event_time + after] (event start time, all High events)."""
        if symbol in self._bcache:
            return self._bcache[symbol]
        e = news.events_for(symbol, self.events)
        w = pd.DataFrame({"start": e.time - pd.Timedelta(minutes=config.NEWS_BLOCK_BEFORE_MIN),
                          "end": e.time + pd.Timedelta(minutes=config.NEWS_BLOCK_AFTER_MIN),
                          "event": e.event.values}).sort_values("start").reset_index(drop=True)
        for c in ["start", "end"]:
            w[c] = pd.to_datetime(w[c], utc=True)
        self._bcache[symbol] = w
        return w

    # ---- windows per pair --------------------------------------------------
    def windows(self, symbol: str) -> pd.DataFrame:
        """DataFrame [event_time, start, end, event] sorted by start (UTC)."""
        if symbol in self._cache:
            return self._cache[symbol]
        e = news.events_for(symbol, self.events)
        p = self.p
        speech = e.duration_min > 0
        before = np.where(speech, p.get("speech_before_min", 10), p.get("news_before_min", 5))
        after = np.where(speech, p.get("speech_after_min", 10), p.get("news_after_min", 5)) + e.duration_min
        w = pd.DataFrame({
            "event_time": e.time.values,
            "start": (e.time - pd.to_timedelta(before, unit="m")).values,
            "end": (e.time + pd.to_timedelta(after, unit="m")).values,
            "event": e.event.values,
        }).sort_values("start").reset_index(drop=True)
        for c in ["event_time", "start", "end"]:
            w[c] = pd.to_datetime(w[c], utc=True)
        self._cache[symbol] = w
        return w

    def _buffer(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.p.get("close_buffer_min", 2))

    # ---- live questions ----------------------------------------------------
    def blocked(self, symbol: str, t: pd.Timestamp) -> str | None:
        """Event name if a NEW trade / pending order is not allowed at time t, else None.
        Blocked from (window start - close buffer) to window end: a trade opened just before
        the window would have to be closed again immediately."""
        if self.block_on:
            b = self.block_windows(symbol)
            hit = b[(b.start <= t) & (t <= b.end)]
            if not hit.empty:
                return str(hit.event.iloc[0])
        if not self.on:
            return None
        w = self.windows(symbol)
        hit = w[(w.start - self._buffer() <= t) & (t <= w.end)]
        return None if hit.empty else str(hit.event.iloc[0])

    def close_deadline(self, symbol: str, opened_at: pd.Timestamp) -> tuple[pd.Timestamp, str] | None:
        """For a trade opened at opened_at: (time it must be closed by, event) or None.
        Only the FIRST upcoming window that is not exempt (event < exempt_hours after opening)."""
        if not self.on:
            return None
        w = self.windows(symbol)
        ex = pd.Timedelta(hours=self.p.get("exempt_hours", 5))
        nxt = w[(w.start > opened_at) & (w.event_time - opened_at < ex)]
        if nxt.empty:
            return None
        r = nxt.iloc[0]
        return r.start - self._buffer(), str(r.event)

    def in_window(self, symbol: str, t: pd.Timestamp) -> bool:
        w = self.windows(symbol)
        return bool(((w.start <= t) & (t <= w.end)).any())

    # ---- backtest ----------------------------------------------------------
    def apply_backtest(self, cands: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict]:
        """cands = portfolio_sim.load_candidates() output (r = GROSS R, exit_time, reason).
        Returns (new cands, stats). Costs are still charged later by portfolio_sim.simulate."""
        stats = dict(profile=self.p["name"], news_rules=self.on, news_block=self.block_on,
                     block_before_min=config.NEWS_BLOCK_BEFORE_MIN, block_after_min=config.NEWS_BLOCK_AFTER_MIN,
                     blocked_fills=0, closed_before_news=0, profit_removed=0, calendar_end=None)
        if not (self.on or self.block_on) or cands.empty:
            return cands, stats
        cal_end = self.events.time.max()
        stats["calendar_end"] = str(cal_end.date())
        out = []
        m5_cache: dict[str, pd.DataFrame] = {}
        for c in cands.itertuples(index=False):
            row = c._asdict()
            if self.blocked(c.symbol, c.fill_time):
                stats["blocked_fills"] += 1
                continue
            dl = self.close_deadline(c.symbol, c.fill_time) if self.on else None
            if dl is not None and c.exit_time > dl[0]:
                if self.p.get("pre_news", "close") == "close":
                    if c.symbol not in m5_cache:
                        m5_cache[c.symbol] = pd.read_parquet(DATA / f"{c.symbol}_M5.parquet", columns=["close", "spread"])
                    m5 = m5_cache[c.symbol]
                    k = m5.index.searchsorted(dl[0], side="right") - 2   # last M5 bar fully closed before deadline
                    k = max(k, int(m5.index.searchsorted(c.fill_time)))  # never before the fill bar
                    px, sp = float(m5.close.iat[k]), float(m5.spread.iat[k])
                    bull = c.direction == "bull"
                    row["r"] = ((px - c.entry) if bull else (c.entry - px - sp)) / c.risk
                    row["exit_time"] = m5.index[k] + pd.Timedelta(minutes=5)
                    row["reason"] = "news"
                    stats["closed_before_news"] += 1
                elif self.in_window(c.symbol, c.exit_time) and c.r > 0:
                    row["r"] = 0.0          # prop firm deducts the profit
                    stats["profit_removed"] += 1
            out.append(row)
        log(f"news ({self.p['name']}, block {'ON' if self.block_on else 'OFF'}): {stats['blocked_fills']} fills blocked, "
            f"{stats['closed_before_news']} closed before news, {stats['profit_removed']} profits removed "
            f"(calendar ends {stats['calendar_end']})")
        return pd.DataFrame(out).reset_index(drop=True), stats


def load_guard(profile_name: str | None = None, log=print) -> NewsGuard:
    return NewsGuard(news.load_events(log=log), config.profile(profile_name))
