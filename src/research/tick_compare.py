"""Research-only tick exit comparison. Never changes live state or model artifacts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config
from src.research.management import POLICIES, run_trade_full
from src.risk.portfolio_sim import PIP, risk_usd_per_unit

POL = {p.name: p for p in POLICIES}


@dataclass
class Outcome:
    status: str
    r: float | None = None
    reason: str | None = None
    filled_ms: int | None = None
    exited_ms: int | None = None


def tick_trade(times, bids, asks, bar_ms, entry, sl, bull, policy, trail_mode="tick", hold_deadline_ms=None,
               fixed_fill_ms=None, fill_quote=None):
    """Entry within the original signal's M5 window; exits use only subsequent ticks.

    Fixed limit/stop level fills isolate timing. Gaps/latency are NOT modeled here.
    bar mode updates trailing only after an M5 bar closes; tick mode updates after
    checking the existing stop/target at each tick. second mode keeps broker
    stops active on every tick but moves trailing from whole-second observations
    only (ideal immediate acceptance, no network delay). Incomplete windows are censored.
    """
    if trail_mode not in ("bar", "tick", "second"):
        raise ValueError("trail_mode must be bar, tick or second")
    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError("risk must be positive")
    if fixed_fill_ms is None:
        indices = np.arange(np.searchsorted(times, bar_ms), np.searchsorted(times, bar_ms + 300_000))
        hits = asks[indices] <= entry if bull else bids[indices] >= entry
        found = indices[hits]
        if not len(found):
            return Outcome("no_tick_fill")
        f = int(found[0])
        filled = int(times[f])
        initial = ()
    else:
        # Exit-only experiment: the order fill is an external fixed event, not
        # rediscovered from slower polls. Its quote handles spread at entry.
        filled = int(fixed_fill_ms)
        if not bar_ms <= filled < bar_ms + 300_000 or fill_quote is None:
            raise ValueError("Fixed entry requires a fill-window timestamp and its Bid/Ask quote")
        bid, ask = map(float, fill_quote)
        if not np.isfinite([bid, ask]).all() or bid <= 0 or ask < bid:
            raise ValueError("Invalid fill quote")
        f = int(np.searchsorted(times, filled, side='right'))
        initial = ((filled, bid, ask),)
    stop, best = -1.0, 0.0
    previous_bar = filled // 300_000
    # A caller with M5 coverage can preserve the 864 actual-bar deadline across
    # weekends. Default retains the earlier prototype's calendar-day horizon.
    deadline = bar_ms + 3 * 86400_000 if hold_deadline_ms is None else hold_deadline_ms
    from itertools import chain
    observations = chain(initial, ((int(times[i]), bids[i], asks[i]) for i in range(f, len(times))))
    observations = ((t,b,a,False) for t,b,a in observations)
    if trail_mode == 'second':
        from heapq import merge
        # Tick events win ties: server stops see all quotes at this timestamp
        # before the bot observes the latest one. Polls never use future quotes.
        last = min(int(times[-1]), int(deadline)-1)
        grid = np.arange(((filled+999)//1000)*1000,last+1,1000,dtype=np.int64)
        ix = np.searchsorted(times,grid,side='right')-1
        polls = ((int(t),bids[j],asks[j],True) for t,j in zip(grid,ix) if j>=0)
        observations = merge(observations,polls,key=lambda e:(e[0],e[3]))
    for timestamp, bid, ask, is_poll in observations:
        if timestamp >= deadline:
            return Outcome("closed", float((bid - entry if bull else entry - ask) / risk),
                           "timeout", filled, timestamp)
        bar = timestamp // 300_000
        if trail_mode == "bar" and bar > previous_bar:
            if policy.trail_r is not None and best >= policy.trail_start_r:
                stop = max(stop, best - policy.trail_r)
        previous_bar = bar
        r = float((bid - entry if bull else entry - ask) / risk)
        if r <= stop + 1e-10:
            reason = "sl" if stop < -0.01 else ("be" if abs(stop) <= 0.01 else "trail")
            return Outcome("closed", stop, reason, filled, timestamp)
        if policy.tp_r is not None and r >= policy.tp_r - 1e-10:
            return Outcome("closed", policy.tp_r, "tp", filled, timestamp)
        # A one-second bot cannot remember an unobserved intrasecond spike.
        if trail_mode != 'second' or is_poll:
            best = max(best, r)
        if (trail_mode == "tick" or (trail_mode == 'second' and is_poll)) and policy.trail_r is not None and best >= policy.trail_start_r:
            stop = max(stop, best - policy.trail_r)
    return Outcome("unresolved_at_data_end", filled_ms=filled)


def bar_trade(m5, signal, cutoff, post_entry=None):
    frame = m5.loc[(m5.index >= signal.fill_time) & (m5.index + pd.Timedelta(minutes=5) <= cutoff)].iloc[:864].copy()
    frame = frame.astype({c:float for c in ("high","low","close","spread")})
    if frame.empty:
        return Outcome("unresolved_at_data_end")
    filled = int(signal.fill_time.timestamp()*1000)
    if post_entry is not None:
        times, bids, asks, actual_fill = post_entry
        if actual_fill is None:
            return Outcome("no_tick_fill")
        filled = actual_fill
        after = (times >= filled) & (times < int((signal.fill_time+pd.Timedelta(minutes=5)).timestamp()*1000))
        if not after.any():
            return Outcome("unresolved_at_data_end", filled_ms=filled)
        first = frame.index[0]
        frame.loc[first,"high"] = bids[after].max()
        frame.loc[first,"low"] = bids[after].min()
        frame.loc[first,"close"] = bids[after][-1]
        # Keep original M5 spread approximation: this variant isolates removal
        # of pre-entry OHLC, rather than silently changing the short-side quote model.
    r, reason, k = run_trade_full(POL["tp1" if signal.exit_ml == "tp1" else "trail1"],
                                 signal.direction == "bull", signal.entry, abs(signal.entry - signal.sl),
                                 *(frame[c].to_numpy() for c in ("high", "low", "close", "spread")),
                                 fill_bar_partial=False)   # this audit compares the ORIGINAL M5 logic vs ticks
    if reason == "timeout" and len(frame) < 864:
        return Outcome("unresolved_at_data_end")
    return Outcome("closed", float(r), reason, filled,
                   int((frame.index[k] + pd.Timedelta(minutes=5)).timestamp() * 1000))


def net_r(symbol, entry, sl, result):
    if result.status != "closed":
        return None
    risk = abs(entry - sl)
    commission = config.BROKER_COMMISSION / (100_000 * risk_usd_per_unit(symbol, entry, risk))
    slip = 0.2 * PIP[symbol] / risk if result.reason in ("sl", "trail", "be") else 0
    return float(result.r - commission - slip)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=ROOT / "data/ticks/sample/eurusd-tick-2026-09-01-2026-09-02.csv")
    parser.add_argument("--symbol", default="EURUSD", choices=("EURUSD", "GBPUSD", "USDJPY"))
    parser.add_argument("--start", default="2026-09-01")
    parser.add_argument("--end", default="2026-09-02")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/tick-comparison-sample")
    parser.add_argument("--clip", action="store_true", help="Use only ticks inside start/end from a larger file")
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    ticks = pd.read_csv(args.ticks)
    if args.clip:
        ticks = ticks[(ticks.timestamp >= int(start.timestamp()*1000)) & (ticks.timestamp < int(end.timestamp()*1000))]
    times = ticks.timestamp.to_numpy(dtype=np.int64)
    bids, asks = ticks.bidPrice.to_numpy(float), ticks.askPrice.to_numpy(float)
    if not len(times) or (np.diff(times) < 0).any() or not np.isfinite(bids).all() or not np.isfinite(asks).all() or (asks < bids).any() or (bids <= 0).any():
        raise ValueError("Invalid/empty ticks")
    if (times < int(start.timestamp()*1000)).any() or (times >= int(end.timestamp()*1000)).any():
        raise ValueError("Tick file must lie within the specified interval")
    m5 = pd.read_parquet(ROOT / f"data/processed/{args.symbol}_M5.parquet")
    tick_bars = pd.Series(bids, index=pd.to_datetime(times, unit="ms", utc=True)).resample("5min").ohlc().dropna()
    joined = m5[["open", "high", "low", "close"]].join(tick_bars, how="inner", lsuffix="_saved", rsuffix="_ticks")
    audit = {"ticks": len(ticks), "first_tick": str(pd.to_datetime(times[0], unit="ms", utc=True)),
             "last_tick": str(pd.to_datetime(times[-1], unit="ms", utc=True)),
             "matched_m5_bars": len(joined), "largest_gap_seconds": float(np.diff(times).max()/1000),
             "median_spread_pips": float(np.median(asks-bids)/PIP[args.symbol]),
             "ohlc_max_abs_difference_pips": {c: float((joined[c+"_saved"]-joined[c+"_ticks"]).abs().max()/PIP[args.symbol]) for c in ("open","high","low","close")}}
    expected = m5.loc[(m5.index >= start) & (m5.index + pd.Timedelta(minutes=5) <= end)]
    audit["stored_m5_bars_in_window"] = len(expected)
    audit["stored_bars_missing_tick_ohlc"] = len(expected.index.difference(tick_bars.index))
    scores = pd.read_parquet(ROOT / "data/research/ml_scores.parquet")
    signals = scores[(scores.symbol.astype(str) == args.symbol) & (scores.zone_tf.astype(str) == "M15") &
                     (scores.touch_number == 1) & scores["keep_0.3"] & (scores.fill_time >= start) & (scores.fill_time < end)]
    before = len(signals)
    signals = signals[config.va_ok(signals.r24_va_pos) & config.act_ok(signals.act_15m)]
    rows = []
    for signal in signals.sort_values("fill_time").itertuples():
        chosen = POL["tp1" if signal.exit_ml == "tp1" else "trail1"]
        variants = {"baseline_m5": bar_trade(m5, signal, end)}
        for name, policy, mode in (("tick_exit_bar_trail", chosen, "bar"), ("tick_exit_tick_trail", chosen, "tick"),
                                   ("counterfactual_always_trail_bar", POL["trail1"], "bar"),
                                   ("counterfactual_always_trail_tick", POL["trail1"], "tick")):
            variants[name] = tick_trade(times, bids, asks, int(signal.fill_time.timestamp()*1000),
                                       signal.entry, signal.sl, signal.direction == "bull", policy, mode)
        variants["post_entry_m5_exit"] = bar_trade(m5,signal,end,
            (times,bids,asks,variants["tick_exit_tick_trail"].filled_ms))
        for name, result in variants.items():
            rows.append(dict(signal_id=f"{args.symbol}:{signal.Index}", zone_time=str(signal.zone_candle_time),
                             signal_time=str(signal.fill_time), symbol=args.symbol, chosen_exit=signal.exit_ml,
                             entry=signal.entry, sl=signal.sl, scenario=name, **vars(result),
                             fill_time=str(pd.to_datetime(result.filled_ms,unit="ms",utc=True)) if result.filled_ms else None,
                             exit_time=str(pd.to_datetime(result.exited_ms,unit="ms",utc=True)) if result.exited_ms else None,
                             net_r=net_r(args.symbol,signal.entry,signal.sl,result)))
    frame = pd.DataFrame(rows)
    summaries = {}
    if len(frame):
        for name, group in frame.groupby("scenario"):
            closed = group[group.status == "closed"]
            wins, losses = closed.net_r[closed.net_r > 0].sum(), -closed.net_r[closed.net_r < 0].sum()
            summaries[name] = {"signals": len(group), "closed": len(closed), "unresolved": int((group.status == "unresolved_at_data_end").sum()),
                               "no_fill": int((group.status == "no_tick_fill").sum()), "net_r_sum": float(closed.net_r.sum()),
                               "win_pct": float((closed.net_r > 0).mean()*100) if len(closed) else None,
                               "pf": float(wins/losses) if losses else None}
    comparisons = []
    paired_summaries = {}
    if len(frame):
        baseline = frame[frame.scenario == "baseline_m5"].set_index("signal_id")
        actual = frame[frame.scenario == "tick_exit_tick_trail"].set_index("signal_id")
        for signal_id in baseline.index:
            a, b = baseline.loc[signal_id], actual.loc[signal_id]
            comparisons.append({"signal_id": signal_id, "signal_time": a.signal_time, "baseline_exit": a.exit_time,
                                "tick_fill": b.fill_time, "tick_exit": b.exit_time,
                                "baseline_reason": a.reason, "tick_reason": b.reason,
                                "net_r_difference": float(b.net_r-a.net_r) if pd.notna(a.net_r) and pd.notna(b.net_r) else None,
                                "tick_exit_minus_baseline_seconds": float((pd.Timestamp(b.exit_time)-pd.Timestamp(a.exit_time)).total_seconds()) if b.exit_time and a.exit_time else None})
        # Closed-only standalone summaries can have different denominators near the cutoff.
        # Also report a common fully-resolved set to avoid comparing unmatched outcomes.
        names = ["baseline_m5", "post_entry_m5_exit", "tick_exit_bar_trail", "tick_exit_tick_trail"]
        common = set(baseline.index)
        for name in names:
            common &= set(frame[(frame.scenario == name) & (frame.status == "closed")].signal_id)
        for name in names:
            group = frame[(frame.scenario == name) & frame.signal_id.isin(common)]
            wins, losses = group.net_r[group.net_r > 0].sum(), -group.net_r[group.net_r < 0].sum()
            paired_summaries[name] = {"matched_closed_signals": len(group), "net_r_sum": float(group.net_r.sum()),
                                     "win_pct": float((group.net_r > 0).mean()*100) if len(group) else None,
                                     "pf": float(wins/losses) if losses else None}
    report = {"symbol": args.symbol, "start":str(start), "end":str(end), "audit":audit,
              "ml_selected_before_va_activity":before, "eligible_signals":len(signals), "scenarios":summaries,
              "timing_comparison": comparisons,
              "common_closed_summaries": paired_summaries,
              "limitations":["Exploratory one-pair sample, not account returns or an independent holdout.",
                "Only signals whose original fill candle is in this file; excludes prior-day open positions and other pairs.",
                "Original ML, VA and activity selection is frozen; no signal/model retraining.",
                "post_entry_m5_exit clips the fill candle to post-fill BID extremes while retaining the original M5 spread approximation.",
                "No news filtering applied here; historical news calendar coverage must be audited before portfolio testing.",
                "Nominal limit/stop fills, $5/lot commission and 0.2-pip stop slippage; gaps/latency not simulated.",
                "Every recorded tick is processed, not a 1-second latest-price poll.",
                "Open outcomes at the data end are censored, not force-closed.",
                "Always-trail rows are counterfactual mechanics checks, not the approved strategy.",
                "Initial prototype covers fixed TP and trailing only; account/risk/news integration remains pending."]}
    args.out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.out/"trade_comparison.csv",index=False)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2))
    text = ["# Tick comparison: exploratory sample", f"\n{args.symbol}: {args.start} to {args.end} UTC",
            f"\n{len(ticks):,} ticks; {len(joined)} M5 bars reconciled. All times below are UTC.",
            "\n## Selected-signal comparison", "\n| Scenario | Closed | Win rate | Net R | PF |", "|---|---:|---:|---:|---:|"]
    for name in ("baseline_m5", "post_entry_m5_exit", "tick_exit_bar_trail", "tick_exit_tick_trail"):
        if name in summaries:
            s = summaries[name]
            text.append(f"| {name} | {s['closed']} | {s['win_pct']} | {s['net_r_sum']:.4f} | {s['pf']} |")
    text.extend(["\n## Exit timing", "\n| Signal | Tick fill | Baseline exit | Tick exit | Tick minus baseline (seconds) |", "|---|---|---|---|---:|"])
    for c in comparisons:
        text.append(f"| {c['signal_time']} | {c['tick_fill']} | {c['baseline_exit']} | {c['tick_exit']} | {c['tick_exit_minus_baseline_seconds']} |")
    text.extend(["\nThe M5 baseline uses the entire fill candle: favorable movement before the actual tick fill can affect its exit. Timing is not simply shifted earlier by switching to ticks.",
                 "\n## Common fully resolved signals (same denominator)", "\n```json", json.dumps(paired_summaries, indent=2), "```",
                 "\nAlways-trail results in summary.json/CSV are counterfactuals, not selected strategy results.", "\n## Limits"])
    text.extend("- "+v for v in report["limitations"])
    (args.out/"README.md").write_text("\n".join(text))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
