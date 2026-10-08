"""Console-only exit timing audit. No reports, model or live-state writes.

Both downloaded months are exploratory, frozen-signal experiments. Actual tick
fills are held fixed across exit variants; this is not an entry polling test.
"""
from dataclasses import replace
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config
from src.features.news import load_events
from src.research.tick_compare import POL, bar_trade, tick_trade, net_r
from src.research.september_tick_replay import sample_one_second, stats
from src.risk import portfolio_sim as ps
from src.risk.news_rules import NewsGuard

NAMES = ['old_full_candle', 'post_entry_m5', 'tick_bar_trail', 'tick_fast_trail',
         'second_bar_trail', 'second_fast_trail', 'server_stops_second_trail']


def replay_month(start_text, end_text, export_offset):
    start, end = pd.Timestamp(start_text, tz='UTC'), pd.Timestamp(end_text, tz='UTC')
    ledger = pd.read_csv(ROOT/'outputs/trade-stats-2019-2026/strategy_7646_trades.csv')
    for c in ['fill_time', 'exit_time']:
        ledger[c] = pd.to_datetime(ledger[c], utc=True)
    assert ledger[(ledger.fill_time < start) & (ledger.exit_time > start)].empty
    candidates = ledger[(ledger.fill_time >= start) & (ledger.fill_time < end)]
    if start_text == '2026-09-01':
        events = pd.read_csv(ROOT/'outputs/news-calendar-september2026/calendar_utc.csv')
        events['time'] = pd.to_datetime(events.time, utc=True)
        news_source = 'separate retrospective September calendar'
    else:
        events = load_events(start=start_text, log=lambda x: None)
        events = events[events.time < end + pd.Timedelta(days=1)]
        news_source = 'existing historical calendar, not independently re-audited here'
    guard = NewsGuard(events, config.profile(), block=True)
    assert not guard.on
    rows, no_fill, blocked_ids = [], [], set()
    quality = {}
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        filename = f'{symbol}_Ticks_{start:%Y.%m.%d}_{end:%Y.%m.%d}.csv'
        d = pd.read_csv(ROOT/'data/ticks/jforex'/filename)
        dt = pd.to_datetime(d['Time (EET)'], format='%Y.%m.%d %H:%M:%S.%f').dt.tz_localize('UTC') - pd.Timedelta(hours=export_offset)
        times = dt.array.as_unit('ms').asi8
        bids, asks = d.Bid.to_numpy(float), d.Ask.to_numpy(float)
        assert np.all(np.diff(times) >= 0)
        assert np.isfinite(bids).all() and np.isfinite(asks).all()
        assert np.all((asks >= bids) & (bids > 0))
        # Restrict all variants to the same observation horizon. Do not use the
        # extra February hour exported by a January UTC+3 range selection.
        keep = (dt >= start) & (dt < end)
        times, bids, asks = times[keep], bids[keep], asks[keep]
        dt = dt[keep]
        seconds, sb, sa = sample_one_second(times, bids, asks)
        m5 = pd.read_parquet(ROOT/f'data/processed/{symbol}_M5.parquet')
        m5 = m5.astype({c: float for c in ['open', 'high', 'low', 'close', 'spread']})
        bars = pd.Series(bids, index=pd.DatetimeIndex(dt)).resample('5min').ohlc().dropna()
        ix = m5.index[(m5.index >= start) & (m5.index < end)]
        assert ix.difference(bars.index).empty
        differences = (bars.loc[ix] - m5.loc[ix, ['open','high','low','close']]).abs()/ps.PIP[symbol]
        assert (differences.median() < .01).all(), 'Export clock does not reconcile with UTC M5'
        quality[symbol] = {'ticks_in_month': len(times), 'missing_m5': 0,
                           'max_ohlc_difference_pips': differences.max().to_dict()}
        updated = m5.copy()
        updated.loc[ix, ['open','high','low','close']] = bars.loc[ix]
        for signal in candidates[candidates.symbol == symbol].itertuples():
            bar_ms = int(signal.fill_time.timestamp()*1000)
            f = int(m5.index.searchsorted(signal.fill_time))
            deadline = int((m5.index[f+863]+pd.Timedelta(minutes=5)).timestamp()*1000) if f+863 < len(m5) else int(end.timestamp()*1000)+1
            args = (times,bids,asks,bar_ms,signal.entry,signal.sl,signal.direction=='bull',POL['tp1' if signal.exit_ml=='tp1' else 'trail1'])
            tick_bar = tick_trade(*args, trail_mode='bar', hold_deadline_ms=deadline)
            if tick_bar.filled_ms is None:
                no_fill.append(signal.trade_id)
                continue
            filled = tick_bar.filled_ms
            same_ms = np.arange(np.searchsorted(times,filled),np.searchsorted(times,filled,side='right'))
            touched = asks[same_ms] <= signal.entry if signal.direction=='bull' else bids[same_ms] >= signal.entry
            k = int(same_ms[touched][0])
            quote = (bids[k], asks[k])
            tick_fast = tick_trade(*args, trail_mode='tick', hold_deadline_ms=deadline)
            old = bar_trade(m5, signal, end)
            assert old.status != 'closed' or (abs(old.r-signal.r)<1e-6 and old.reason==signal.reason)
            # Old OHLC outcome retained only as a diagnostic reference. Give it
            # the same actual entry time for news, sizing and account ordering.
            results = {'old_full_candle': replace(old, filled_ms=filled),
                       'post_entry_m5': bar_trade(updated,signal,end,(times,bids,asks,filled)),
                       'tick_bar_trail': tick_bar, 'tick_fast_trail': tick_fast,
                       'server_stops_second_trail': tick_trade(*args,trail_mode='second',hold_deadline_ms=deadline)}
            for name, mode in [('second_bar_trail','bar'),('second_fast_trail','tick')]:
                results[name] = tick_trade(seconds,sb,sa,bar_ms,signal.entry,signal.sl,
                    signal.direction=='bull',args[-1],trail_mode=mode,
                    hold_deadline_ms=deadline,fixed_fill_ms=filled,fill_quote=quote)
            assert all(x.filled_ms == filled for x in results.values())
            fill_time = pd.to_datetime(filled, unit='ms', utc=True)
            blocked = guard.blocked(symbol, fill_time) is not None
            if blocked:
                blocked_ids.add(signal.trade_id)
            for name, x in results.items():
                rows.append(dict(trade_id=signal.trade_id,symbol=symbol,direction=signal.direction,
                    entry=signal.entry,risk=abs(signal.entry-signal.sl),exit=signal.exit_ml,r=x.r,reason=x.reason,
                    scenario=name,status=x.status,fill_time=fill_time,
                    exit_time=pd.to_datetime(x.exited_ms,unit='ms',utc=True) if x.exited_ms is not None else pd.NaT,
                    net_r=net_r(symbol,signal.entry,signal.sl,x),news_blocked=blocked))
        print(f'{start:%Y-%m} {symbol}: fixed-entry exits checked',flush=True)
    frame = pd.DataFrame(rows)
    # Compare identical fully observed signals. A censored trade requires an
    # exit tail and is never silently booked as a zero or forced month-end exit.
    common = set(frame.trade_id)
    for name in NAMES:
        common &= set(frame[(frame.scenario==name)&(frame.status=='closed')&~frame.news_blocked].trade_id)
    eligible = set(frame[~frame.news_blocked].trade_id)
    excluded = sorted(eligible-common)
    summaries, accounts = {}, {}
    for name in NAMES:
        g = frame[(frame.scenario==name)&frame.trade_id.isin(common)].copy()
        summaries[name] = stats(g)
        g = g.sort_values(['fill_time','trade_id'],kind='stable').reset_index(drop=True)
        res = ps.simulate(g,ps.combo_config(10000,5))
        accounts[name] = {k:res[k] for k in ['trades','final_equity','profit_factor','win_pct','max_drawdown_pct']}
        accounts[name]['profit_usd'] = round(res['final_equity']-10000,2)
    base = frame[frame.scenario=='tick_fast_trail'].set_index('trade_id')
    differences = []
    for row in frame[(frame.scenario=='second_fast_trail')&frame.trade_id.isin(common)].itertuples():
        a = base.loc[row.trade_id]
        if abs(a.r-row.r)>1e-8 or a.reason!=row.reason:
            differences.append({'trade_id':row.trade_id,'symbol':row.symbol,
                'tick_r':a.r,'second_r':row.r,'tick_reason':a.reason,'second_reason':row.reason})
    result = dict(period=f'{start:%Y-%m}',selected_signals=len(candidates),
        no_actual_tick_fill=no_fill,news_blocked=len(blocked_ids),common_signals=len(common),
        censored_excluded=excluded,quality=quality,news_source=news_source,
        signal_stats=summaries,fresh_account=accounts,tick_vs_second_fast_changes=differences)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    replay_month('2019-01-01','2019-02-01',2)
    replay_month('2026-09-01','2026-10-01',3)
