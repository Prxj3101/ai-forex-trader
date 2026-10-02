"""
News filter test on the BOT's trades (M15 touch 1 + ML top 30% + exit brain, NET of costs).

Question: do trades near high-impact news (NFP, FOMC, CPI, BoE, ECB, BOJ ...) do worse?
  - fill timing vs news (minutes before / after the nearest event of the pair's currencies)
  - trades that are OPEN while news comes out ("held through news")
  - block-window variants: no new fills from X min before to Y min after news
  - "flat before news": close open trades N min before news (replayed on M5)

Calendar ends 2025-04-04 -> test period 2019-01 .. 2025-04.
Run: python -m src.research.news_test
Output: data/research/news_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import news
from src.risk import portfolio_sim as ps

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "data" / "research"
DATA = ROOT / "data" / "processed"
CACHE = RESEARCH / "news_candidates.parquet"


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return dict(trades=0, win_pct=None, pf=None, total_r=0.0, avg_r=None)
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=int(len(r)), win_pct=round(float((r > 0).mean() * 100), 1),
                pf=round(float(g / l), 3) if l > 0 else None,
                total_r=round(float(r.sum()), 1), avg_r=round(float(r.mean()), 3))


def load_trades(log=print) -> pd.DataFrame:
    scores = pd.read_parquet(RESEARCH / "ml_scores.parquet")
    for c in ["symbol", "zone_tf", "direction"]:
        scores[c] = scores[c].astype(str)
    if CACHE.exists() and CACHE.stat().st_mtime > (RESEARCH / "ml_scores.parquet").stat().st_mtime:
        c = pd.read_parquet(CACHE)
    else:
        log("replaying bot trades on M5 for exit times ...")
        c = ps.load_candidates(scores=scores)
        c.to_parquet(CACHE)
    key = ["symbol", "direction", "fill_time", "entry"]
    net = scores[(scores.zone_tf == "M15") & (scores.touch_number == 1) & scores["keep_0.3"]][key + ["r_ml"]]
    t = c.merge(net.drop_duplicates(key), on=key, how="left")
    return t


def tag(t: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["min_to_next"] = np.nan    # minutes from fill to next news (>0)
    t["min_since_prev"] = np.nan  # minutes since previous news (>=0)
    t["news_while_open"] = 0
    t["first_news_in_trade"] = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns, UTC]")
    for sym, g in t.groupby("symbol"):
        e = news.events_for(sym, ev).time.sort_values().values
        f = g.fill_time.values
        x = g.exit_time.values
        i_next = np.searchsorted(e, f, side="right")
        nxt = np.where(i_next < len(e), e[np.minimum(i_next, len(e) - 1)], np.datetime64("NaT"))
        prv = np.where(i_next > 0, e[np.maximum(i_next - 1, 0)], np.datetime64("NaT"))
        t.loc[g.index, "min_to_next"] = (nxt - f) / np.timedelta64(1, "m")
        t.loc[g.index, "min_since_prev"] = (f - prv) / np.timedelta64(1, "m")
        i_exit = np.searchsorted(e, x, side="left")
        t.loc[g.index, "news_while_open"] = (i_exit - i_next).clip(0)
        t.loc[g.index, "first_news_in_trade"] = pd.DatetimeIndex(
            np.where(i_exit > i_next, nxt, np.datetime64("NaT"))).tz_localize("UTC")
    return t


def flat_before_news(t: pd.DataFrame, minutes: int) -> pd.Series:
    """Net R if trades open over news are closed at the M5 close `minutes` before the news."""
    out = t.r_ml.copy()
    m = t.first_news_in_trade.notna()
    for sym, g in t[m].groupby("symbol"):
        m5 = pd.read_parquet(DATA / f"{sym}_M5.parquet", columns=["close", "spread"])
        idx = m5.index
        for i, r in g.iterrows():
            cut = r.first_news_in_trade - pd.Timedelta(minutes=minutes)
            if cut <= r.fill_time:
                continue  # filled right before news: keep (handled by the block window)
            k = idx.searchsorted(cut, side="right") - 2   # last fully closed bar before cut
            px, sp = m5.close.iat[k], m5.spread.iat[k]
            bull = r.direction == "bull"
            gross = ((px - r.entry) if bull else (r.entry - px - sp)) / r.risk
            # cost in R as in walkforward.add_costs: r_ml (net) - r (gross) when it's a full trade
            cost = (r.r - r.r_ml) if pd.notna(r.r_ml) else 0.0
            out.at[i] = gross - max(cost, 0.0)
    return out


def run(log=print) -> dict:
    ev = news.load_events(log=log)
    t = load_trades(log)
    end = ev.time.max() + pd.Timedelta(days=1)
    t = t[(t.fill_time < end) & t.r_ml.notna()].reset_index(drop=True)
    t = tag(t, ev)
    log(f"bot trades with news coverage: {len(t):,} ({t.fill_time.min():%Y-%m} .. {t.fill_time.max():%Y-%m})")

    res: dict = dict(period=f"{t.fill_time.min():%Y-%m-%d} .. {t.fill_time.max():%Y-%m-%d}",
                     all=stats(t.r_ml))

    # 1) fill timing buckets
    b = pd.Series("far from news", index=t.index)
    b[t.min_to_next <= 120] = "60-120m before"
    b[t.min_to_next <= 60] = "30-60m before"
    b[t.min_to_next <= 30] = "0-30m before"
    b[t.min_since_prev < 120] = "60-120m after"
    b[t.min_since_prev < 60] = "30-60m after"
    b[t.min_since_prev < 30] = "0-30m after"
    res["fill_timing"] = {k: stats(t.r_ml[b == k]) for k in
                          ["0-30m before", "30-60m before", "60-120m before",
                           "0-30m after", "30-60m after", "60-120m after", "far from news"]}

    # 2) held through news
    res["held_through"] = {"no news while open": stats(t.r_ml[t.news_while_open == 0]),
                           "news while open": stats(t.r_ml[t.news_while_open > 0])}

    # 3) event types (trades open over the news, by first event name)
    evmap = ev.set_index(["time"]).event
    held = t[t.first_news_in_trade.notna()].copy()
    held["ev"] = held.first_news_in_trade.map(lambda x: evmap.loc[x] if not isinstance(evmap.loc[x], pd.Series)
                                              else evmap.loc[x].iloc[0])
    def group(s: str) -> str:
        for k, n in [("Non-Farm", "NFP"), ("FOMC", "FOMC"), ("Federal Funds", "FOMC"), ("CPI", "CPI"),
                     ("Retail Sales", "Retail sales"), ("Unemployment Claims", "Jobless claims"),
                     ("Powell", "Fed chair"), ("ISM", "ISM PMI"), ("PMI", "Flash PMI"),
                     ("Bank Rate", "BoE"), ("Monetary Policy Summary", "BoE"), ("BOE", "BoE"),
                     ("Refinancing", "ECB"), ("ECB", "ECB"), ("BOJ", "BOJ"), ("Monetary Policy Statement", "ECB/BOJ"),
                     ("GDP", "GDP"), ("Election", "Election")]:
            if k in s:
                return n
        return "Other"
    held["grp"] = held.ev.map(group)
    res["held_by_event"] = {k: stats(g.r_ml) for k, g in held.groupby("grp")}

    # 4) block windows (no new fills inside [news - before, news + after])
    blocks = {}
    for before in [0, 15, 30, 60]:
        for after in [0, 15, 30, 60]:
            if before == after == 0:
                continue
            bad = (t.min_to_next <= before) | (t.min_since_prev < after)
            blocks[f"{before}m before / {after}m after"] = dict(**stats(t.r_ml[~bad]), blocked=int(bad.sum()),
                                                                blocked_stats=stats(t.r_ml[bad]))
    res["block_windows"] = blocks

    # 5) flat before news (close open trades before news)
    flat = {}
    for m in [5, 15, 30]:
        r = flat_before_news(t, m)
        flat[f"close {m}m before"] = stats(r)
        t[f"r_flat{m}"] = r
    res["flat_before_news"] = flat

    # 6) yearly for the main candidates
    t["year"] = t.fill_time.dt.year
    bad30 = (t.min_to_next <= 30) | (t.min_since_prev < 30)
    res["by_year"] = {str(y): dict(all=stats(g.r_ml), block30=stats(g.r_ml[~bad30.loc[g.index]]))
                      for y, g in t.groupby("year")}
    res["by_symbol"] = {s: dict(all=stats(g.r_ml), block30=stats(g.r_ml[~bad30.loc[g.index]]),
                                held=stats(g.r_ml[g.news_while_open > 0]))
                        for s, g in t.groupby("symbol")}
    t.to_parquet(RESEARCH / "news_trades.parquet")
    (RESEARCH / "news_test.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ["all", "fill_timing", "held_through", "held_by_event"]}, indent=1))
    for k, v in r["block_windows"].items():
        print(f"{k:24s} PF {v['pf']}  trades {v['trades']}  R {v['total_r']}  | blocked {v['blocked']} PF {v['blocked_stats']['pf']}")
    print(json.dumps(r["flat_before_news"], indent=1))
