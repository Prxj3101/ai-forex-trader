"""Audit partial tick outcomes against the existing full-history combo replay.

This is a sensitivity experiment replacing only available EURUSD outcomes;
it is not a new tick backtest of all pairs/history and does not modify production.
"""
from pathlib import Path
import inspect
import json
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import config
from src.risk import portfolio_sim as ps
from src.risk.news_rules import load_guard, NewsGuard


def main():
    out=ROOT/'outputs/tick-account-audit'
    out.mkdir(parents=True,exist_ok=True)
    stored=json.loads((ROOT/'data/research/portfolio_sim.json').read_text())
    ledger=pd.read_csv(ROOT/'outputs/trade-stats-2019-2026/strategy_7646_trades.csv')
    ledger['fill_time']=pd.to_datetime(ledger.fill_time,utc=True)
    cands=ps.load_candidates()
    cands=cands[config.va_ok(cands.va_pos)&config.act_ok(cands.act)].reset_index(drop=True)
    keys=['symbol','fill_time','entry','direction']
    cands=cands.merge(ledger[keys+['trade_id']],on=keys,validate='one_to_one',how='left')
    if cands.trade_id.isna().any():
        raise ValueError('Unmatched strategy identities')
    guard=load_guard(log=lambda *a:None)
    # One externally verified event; NOT a complete September calendar.
    event=pd.DataFrame([{'time':pd.Timestamp('2026-09-04 08:30',tz='America/New_York').tz_convert('UTC'),
                         'currency':'USD','event':'Employment Situation (NFP)',
                         'source':'BLS official 2026 schedule','duration_min':0}])
    augmented=NewsGuard(pd.concat([guard.events,event],ignore_index=True),config.profile())
    comparisons=pd.concat([pd.read_csv(ROOT/f'outputs/tick-comparison-{p}-partial/trade_comparison.csv')
                           for p in ['september','january2019']],ignore_index=True)
    comparisons['signal_time']=pd.to_datetime(comparisons.signal_time,utc=True)
    source=inspect.getsource(ps.simulate)
    needle='open_pos.append(dict(symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,'
    assert needle in source
    source=source.replace(needle,'open_pos.append(dict(trade_id=c.trade_id, entry=c.entry, risk=c.risk, symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,')
    source=source.replace('skipped[why] = skipped.get(why, 0) + 1',
                          'skipped[why] = skipped.get(why, 0) + 1\n        decisions.append(dict(trade_id=c.trade_id, status="skipped", reason=why))')
    summaries={}
    replay_trades={}
    replay_skips={}
    original_report=ps.report

    def replay(name,frame,news_guard):
        decisions=[]
        captured={}
        def capture(closed,curve,skipped,cfg,n):
            captured['closed']=pd.DataFrame(closed)
            return original_report(closed,curve,skipped,cfg,n)
        namespace=dict(vars(ps),decisions=decisions,report=capture)
        exec(compile(source,'research_only_instrumented_simulation','exec'),namespace)
        filtered,news_stats=news_guard.apply_backtest(frame,log=lambda *a:None)
        res=namespace['simulate'](filtered,ps.combo_config(stored['config']['capital'],stored['config']['commission_per_lot']))
        summaries[name]={k:res[k] for k in ['trades','final_equity','profit_factor','win_pct','max_drawdown_pct','skipped']}
        summaries[name]['news']=news_stats
        replay_trades[name]=captured['closed']
        replay_skips[name]={d['trade_id']:d['reason'] for d in decisions}
        captured['closed'].to_csv(out/f'{name}_account.csv',index=False)

    replay('existing_m5',cands,guard)
    for key in ['trades','final_equity','profit_factor','win_pct','max_drawdown_pct']:
        assert summaries['existing_m5'][key]==stored[key],(key,summaries['existing_m5'][key],stored[key])
    def replace_outcomes(scenario):
        frame=cands.copy()
        for row in comparisons[comparisons.scenario==scenario].itertuples():
            if row.status!='closed':
                continue
            match=(frame.symbol==row.symbol)&(frame.fill_time==row.signal_time)&((frame.entry-row.entry).abs()<1e-9)
            if match.sum()!=1:
                raise ValueError('Ambiguous tick outcome identity')
            frame.loc[match,'r']=row.r
            frame.loc[match,'reason']=row.reason
            frame.loc[match,'fill_time']=pd.Timestamp(row.fill_time)
            frame.loc[match,'exit_time']=pd.Timestamp(row.exit_time)
        return frame.sort_values('fill_time',kind='stable').reset_index(drop=True)
    # Outcomes and entry/exit timestamps change; all other candidates keep M5 logic.
    replay('partial_ticks_bar_trail',replace_outcomes('tick_exit_bar_trail'),guard)
    replay('partial_ticks_tick_trail',replace_outcomes('tick_exit_tick_trail'),guard)
    replay('m5_plus_verified_nfp',cands,augmented)
    replay('partial_ticks_plus_verified_nfp',replace_outcomes('tick_exit_tick_trail'),augmented)
    old=comparisons[comparisons.scenario=='baseline_m5'].set_index('signal_id')
    new=comparisons[comparisons.scenario=='tick_exit_tick_trail'].set_index('signal_id')
    rows=[]
    for sid in old.index[(old.r-new.r).abs()>1e-8]:
        a,b=old.loc[sid],new.loc[sid]
        candidate=cands[(cands.symbol==a.symbol)&(cands.fill_time==a.signal_time)&((cands.entry-a.entry).abs()<1e-9)].iloc[0]
        row={'signal_id':sid,'trade_id':candidate.trade_id,'symbol':a.symbol,'signal_time':str(a.signal_time),
             'baseline_gross_r':a.r,'tick_gross_r':b.r,
             'historical_news_coverage':bool(candidate.fill_time<=guard.events.time.max()),
             'existing_calendar_block':guard.blocked(a.symbol,candidate.fill_time),
             'verified_nfp_block':augmented.blocked(a.symbol,pd.Timestamp(b.fill_time))}
        for name,closed in replay_trades.items():
            taken=closed[closed.trade_id==candidate.trade_id]
            row[name+'_taken']=len(taken)==1
            row[name+'_pnl']=float(taken.pnl.iloc[0]) if len(taken)==1 else None
            row[name+'_skip']=replay_skips[name].get(candidate.trade_id)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'changed_trade_audit.csv',index=False)
    report={'baseline_reconciles':True,'historical_calendar_end':str(guard.events.time.max()),
            'verified_event_source':'https://www.bls.gov/schedule/2026/09_sched.htm',
            'changed_trades':rows,'sensitivity_replays':summaries,
            'limits':['Only 16 EURUSD selected outcomes/timestamps are replaced; all other entries/exits remain M5.',
                      'Replay starts at original $10k and includes all three pairs and shared combo/daily/position rules.',
                      'One verified NFP event is added in research memory only; the September calendar is still incomplete.',
                      'New trades unlocked by changed timing have M5 outcomes unless available in the frozen tick sample.',
                      'Frozen ML/VA/activity selection, nominal fills, existing cost model; not a complete live execution replay.',
                      'Final balances are limited sensitivity experiments, not new validated full-history tick returns.']}
    (out/'audit.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str))


if __name__=='__main__':
    main()
