"""Frozen-signal September tick mechanics and fresh-account comparison.

Research artifacts only. No production source, models, live state or report writes.
"""
from pathlib import Path
import inspect
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import config
from src.research.tick_compare import POL, bar_trade, tick_trade, net_r
from src.risk import portfolio_sim as ps
from src.risk.news_rules import NewsGuard

START=pd.Timestamp('2026-09-01',tz='UTC')
END=pd.Timestamp('2026-10-01',tz='UTC')
NAMES=['original_m5','post_entry_m5','tick_exit_bar_trail','tick_exit_tick_trail',
       'one_second_bar_trail','one_second_fast_trail']
OUT=ROOT/'outputs/september-tick-replay'


def stats(frame):
    r=frame.net_r.dropna()
    gains=float(r[r>0].sum()); losses=float(-r[r<0].sum())
    return {'signals':len(frame),'closed':len(r),'net_r':float(r.sum()),
            'pf':gains/losses if losses else None,'win_pct':float((r>0).mean()*100) if len(r) else None}


def sample_one_second(times,bids,asks):
    """Latest quote available at each whole UTC second; never read future ticks.

    Retain repeated observations: clock-based trailing updates can matter even
    when the quote has not changed.
    """
    grid=np.arange(((int(times[0])+999)//1000)*1000,int(times[-1])+1,1000,dtype=np.int64)
    ix=np.searchsorted(times,grid,side='right')-1
    valid=ix>=0
    grid,ix=grid[valid],ix[valid]
    return grid,bids[ix],asks[ix]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ledger=pd.read_csv(ROOT/'outputs/trade-stats-2019-2026/strategy_7646_trades.csv')
    for c in ['fill_time','exit_time']:
        ledger[c]=pd.to_datetime(ledger[c],utc=True)
    prior=ledger[(ledger.fill_time<START)&(ledger.exit_time>START)]
    assert prior.empty, 'Selected carry-in positions require a stateful research replay'
    candidates=ledger[(ledger.fill_time>=START)&(ledger.fill_time<END)].copy()
    events=pd.read_csv(ROOT/'outputs/news-calendar-september2026/calendar_utc.csv')
    events['time']=pd.to_datetime(events.time,utc=True)
    guard=NewsGuard(events,config.profile(),block=True)
    assert not guard.on, 'Evaluation profile expected; compliance early-close requires additional replay'
    rows=[]
    for symbol in ['EURUSD','GBPUSD','USDJPY']:
        d=pd.read_csv(ROOT/f'data/ticks/jforex/{symbol}_Ticks_2026.09.01_2026.10.01.csv')
        dt=pd.to_datetime(d['Time (EET)'],format='%Y.%m.%d %H:%M:%S.%f').dt.tz_localize('UTC')-pd.Timedelta(hours=3)
        times=dt.array.as_unit('ms').asi8
        bids=d.Bid.to_numpy(float); asks=d.Ask.to_numpy(float)
        seconds,second_bids,second_asks=sample_one_second(times,bids,asks)
        assert np.all(np.diff(times)>=0) and np.all(asks>=bids)
        assert pd.to_datetime(times[0],unit='ms',utc=True)==dt.iloc[0], 'Timestamp unit conversion failed'
        m5=pd.read_parquet(ROOT/f'data/processed/{symbol}_M5.parquet').copy()
        m5=m5.astype({c:float for c in ['open','high','low','close','spread']})
        jbars=pd.Series(bids,index=pd.DatetimeIndex(dt)).resample('5min').ohlc().dropna()
        new=m5.copy(); ix=m5.index[(m5.index>=START)&(m5.index<END)]
        assert not len(ix.difference(jbars.index))
        new.loc[ix,['open','high','low','close']]=jbars.loc[ix,['open','high','low','close']]
        for signal in candidates[candidates.symbol==symbol].sort_values('fill_time').itertuples():
            bar_ms=int(signal.fill_time.timestamp()*1000)
            f=int(m5.index.searchsorted(signal.fill_time))
            # Preserve the existing 864 stored-bar maximum, including weekends.
            deadline=int((m5.index[f+863]+pd.Timedelta(minutes=5)).timestamp()*1000) if f+863<len(m5) else int(END.timestamp()*1000)+1
            a=tick_trade(times,bids,asks,bar_ms,signal.entry,signal.sl,signal.direction=='bull',
                         POL['tp1' if signal.exit_ml=='tp1' else 'trail1'],'bar',hold_deadline_ms=deadline)
            b=tick_trade(times,bids,asks,bar_ms,signal.entry,signal.sl,signal.direction=='bull',
                         POL['tp1' if signal.exit_ml=='tp1' else 'trail1'],'tick',hold_deadline_ms=deadline)
            results={'original_m5':bar_trade(m5,signal,END),
                     'post_entry_m5':bar_trade(new,signal,END,(times,bids,asks,b.filled_ms)),
                     'tick_exit_bar_trail':a,'tick_exit_tick_trail':b}
            for name,mode in [('one_second_bar_trail','bar'),('one_second_fast_trail','tick')]:
                results[name]=tick_trade(seconds,second_bids,second_asks,bar_ms,signal.entry,signal.sl,
                                        signal.direction=='bull',POL['tp1' if signal.exit_ml=='tp1' else 'trail1'],
                                        mode,hold_deadline_ms=deadline)
            old=results['original_m5']
            if old.status=='closed':
                assert abs(old.r-signal.r)<1e-6 and old.reason==signal.reason,(signal.trade_id,old,signal.r)
            for name,result in results.items():
                filled=pd.to_datetime(result.filled_ms,unit='ms',utc=True) if result.filled_ms is not None else None
                exited=pd.to_datetime(result.exited_ms,unit='ms',utc=True) if result.exited_ms is not None else None
                blocked=guard.blocked(symbol,filled) if filled is not None else None
                rows.append({'trade_id':signal.trade_id,'symbol':symbol,'direction':signal.direction,
                             'signal_time':str(signal.fill_time),'entry':signal.entry,'sl':signal.sl,
                             'risk':abs(signal.entry-signal.sl),'exit':signal.exit_ml,'scenario':name,
                             'status':result.status,'r':result.r,'reason':result.reason,
                             'fill_time':filled,'exit_time':exited,
                             'net_r':net_r(symbol,signal.entry,signal.sl,result),
                             'news_blocked':blocked is not None,'news_reason':blocked})
        print(f'{symbol}: tick outcomes complete',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'signal_outcomes.csv',index=False)
    ids=set(candidates.trade_id)
    for name in NAMES:
        ids &= set(frame[(frame.scenario==name)&(frame.status=='closed')].trade_id)
    matched=frame[frame.trade_id.isin(ids)&~frame.news_blocked]
    # Common unblocked IDs keep identical denominators in mechanics comparisons.
    clean=set(ids)
    for name in NAMES:
        clean &= set(matched[matched.scenario==name].trade_id)
    common={name:stats(frame[(frame.scenario==name)&frame.trade_id.isin(clean)]) for name in NAMES}
    source=inspect.getsource(ps.simulate)
    needle='open_pos.append(dict(symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,'
    assert needle in source
    source=source.replace(needle,'open_pos.append(dict(trade_id=c.trade_id, entry=c.entry, risk=c.risk, symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,')
    accounts={}
    for name in NAMES:
        g=frame[(frame.scenario==name)&(frame.status=='closed')&~frame.news_blocked].copy()
        g=g.sort_values(['fill_time','trade_id'],kind='stable').reset_index(drop=True)
        captured={}
        def capture(closed,curve,skipped,cfg,n):
            captured['closed']=pd.DataFrame(closed)
            return ps.report(closed,curve,skipped,cfg,n)
        namespace=dict(vars(ps),report=capture)
        exec(compile(source,'research_september_account','exec'),namespace)
        res=namespace['simulate'](g,ps.combo_config(10000,5))
        closed=captured['closed']
        closed.to_csv(OUT/f'{name}_account.csv',index=False)
        account={k:res[k] for k in ['trades','final_equity','profit_factor','win_pct','max_drawdown_pct','worst_day_pct','commission_usd','skipped','risk_used','combo','kill_switch_events']}
        account['profit_usd']=round(float(closed.pnl.sum()),2)
        account['return_pct']=float(closed.pnl.sum()/10000*100)
        account['news_blocked_signals']=int(frame.loc[frame.scenario==name,'news_blocked'].sum())
        account['no_fill_signals']=int(((frame.scenario==name)&(frame.status=='no_tick_fill')).sum())
        account['censored_signals']=int(((frame.scenario==name)&(frame.status=='unresolved_at_data_end')).sum())
        accounts[name]=account
    baseline=frame[frame.scenario=='original_m5'].set_index('trade_id')
    changes=[]
    for row in frame[frame.scenario!='original_m5'].itertuples():
        old=baseline.loc[row.trade_id]
        both=row.status=='closed' and old.status=='closed'
        changes.append({'trade_id':row.trade_id,'symbol':row.symbol,'scenario':row.scenario,
                        'old_status':old.status,'new_status':row.status,
                        'old_r':old.r,'new_r':row.r,'old_reason':old.reason,'new_reason':row.reason,
                        'news_blocked':row.news_blocked,
                        'r_or_reason_changed':bool(both and (abs(old.r-row.r)>1e-8 or old.reason!=row.reason)),
                        'exit_seconds_difference':float((row.exit_time-old.exit_time).total_seconds()) if both else None})
    pd.DataFrame(changes).to_csv(OUT/'changed_outcomes.csv',index=False)
    report={'research_only':True,'period':'September 2026 UTC','selected_signals':len(candidates),
            'common_closed_unblocked_signals':len(clean),'common_signal_stats':common,
            'fresh_10000_account':accounts,
            'change_counts':{name:sum(c['r_or_reason_changed'] and not c['news_blocked'] for c in changes if c['scenario']==name) for name in NAMES[1:]},
            'limits':['Frozen 79 selected entries/ML exit choices/VA/activity; intrabar zones and score updates are not tested.',
                      'All account variants start September with a fresh $10,000, aggressive combo, no prior-month drawdown or positions; not a continuation of the 2019 account.',
                      'Original selected ledger has no September carry-in positions; censored outcomes excluded rather than force-closed.',
                      'tick_* processes every recorded quote (ideal); one_second_* observes only the latest available quote at whole UTC seconds. Actual poll phase, outages and broker latency are not modeled.',
                      'Nominal limit/stop fills, $5 per lot roundtrip, 0.2-pip stop slippage; gap/latency/fill-quality stress still pending.',
                      'M5 short exits retain stored spread approximation; tick short exits use actual Ask. That quote-model change is included in the comparison.',
                      'News uses separate retrospective September provider calendar with -5/+20 entry block, not historical live-calendar snapshots.',
                      'Drawdown is CLOSED balance only, not intratrade floating equity drawdown; no annualized-return claim from one month.',
                      'Research is exploratory, not independent holdout evidence or approval to deploy. Production bot, models, live state and reports unchanged.']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    labels={'original_m5':'Old M5 + corrected news','post_entry_m5':'Post-entry M5','tick_exit_bar_trail':'Tick SL/TP + M5 trail','tick_exit_tick_trail':'Tick SL/TP + tick trail',
            'one_second_bar_trail':'1s SL/TP + M5 trail','one_second_fast_trail':'1s SL/TP + 1s trail'}
    text='# September tick comparison — research only\n\nAll variants: fresh $10,000, approved combo risk rules, corrected September news blocks.\n\n'
    text+='| Scenario | Trades | Profit USD | Return | PF | Win | Closed DD |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,r in accounts.items():
        text+=f"| {labels[name]} | {r['trades']} | {r['profit_usd']:.2f} | {r['return_pct']:.2f}% | {r['profit_factor']} | {r['win_pct']}% | {r['max_drawdown_pct']}% |\n"
    text+='\nOriginal M5 is also news-corrected for a fair timing comparison. This table is September only, not the existing multi-year account. Post-entry M5 isolates removal of movement before entry; subsequent tick variants add event ordering/actual quotes/trailing timing.\n\n'
    text+='## Common closed signals before account constraints\n\n```json\n'+json.dumps(common,indent=2)+'\n```\n\n## Limits\n\n'
    text+='\n'.join('- '+x for x in report['limits'])+'\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
