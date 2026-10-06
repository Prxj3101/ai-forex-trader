from pathlib import Path
import sys, json, hashlib, inspect
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.risk import portfolio_sim as ps

OUT = Path(__file__).parent
source_paths = [ROOT/'data/research/ml_scores.parquet', ROOT/'data/research/ml_walkforward.json', ROOT/'data/research/portfolio_sim.json']
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
scores = pd.read_parquet(source_paths[0])
strategy = scores[(scores.zone_tf == 'M15') & (scores.touch_number == 1) & scores['keep_0.3'] & scores.va_ok & scores.act_ok].sort_values('fill_time').reset_index(drop=True)
assert len(strategy) == 7646
print('Replaying original exits without training or saving models...', flush=True)
candidates = ps.load_candidates(scores=scores)
captured = {}
original_report = ps.report
def capture(closed, curve, skipped, cfg, n_cands):
    captured['closed'] = pd.DataFrame(closed)
    captured['curve'] = curve
    return original_report(closed, curve, skipped, cfg, n_cands)
ps.report = capture
# Preserve entry/stop identity in the exported ledger only. All simulation logic
# remains identical; multiple zones can fill at the same timestamp.
simulation_source = inspect.getsource(ps.simulate)
needle = 'open_pos.append(dict(symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,'
assert needle in simulation_source
simulation_source = simulation_source.replace(needle, 'open_pos.append(dict(entry=c.entry, risk=c.risk, symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,')
namespace = dict(vars(ps))
exec(compile(simulation_source, 'report_only_simulation', 'exec'), namespace)
ps.simulate = namespace['simulate']
stored = json.loads(source_paths[2].read_text())
sim = ps.run(capital=stored['config']['capital'], risk_pct=stored['config']['risk_pct'], commission=stored['config']['commission_per_lot'], min_sl_pips=stored['config']['min_sl_pips'], cands=candidates, profile_name=stored['account_profile']['name'], log=lambda *a, **k: None, combo=stored['config'].get('combo_enabled', False))
for key in ['trades', 'candidates', 'final_equity', 'total_return_pct', 'cagr_pct', 'max_drawdown_pct', 'win_pct', 'profit_factor', 'commission_usd', 'yearly_return_pct', 'skipped']:
    assert sim[key] == stored[key], (key, sim[key], stored[key])
account = captured['closed'].sort_values('exit_time', kind='stable').reset_index(drop=True)
strategy['risk'] = (strategy.entry - strategy.sl).abs()
strategy['exit'] = strategy.exit_ml
keys = ['symbol','direction','fill_time','entry','risk','exit']
assert not strategy.duplicated(keys).any()
assert not candidates.duplicated(keys).any()
strategy = strategy.merge(candidates[keys+['exit_time','r','reason']], on=keys, how='left', validate='one_to_one')
assert strategy.exit_time.notna().all()
strategy['trade_id'] = [f'S{i+1:05d}' for i in range(len(strategy))]
strategy['net_r'] = strategy.r_ml
strategy['cost_r'] = strategy.r - strategy.net_r
assert strategy.cost_r.min() >= -1e-8
account = account.merge(strategy[['trade_id']+keys+['exit_time','net_r','score']], on=keys+['exit_time'], how='left', validate='one_to_one')
assert account.trade_id.notna().all()
account['commission_usd'] = account.lots * stored['config']['commission_per_lot']
account['slippage_usd'] = account.r * account.risk_usd - account.commission_usd - account.pnl
account['effective_net_r'] = account.pnl / account.risk_usd
account['net_r_replay_difference'] = account.effective_net_r - account.net_r
print('Stored strategy R versus replay account R: differing trades', int((account.net_r_replay_difference.abs()>1e-7).sum()), 'max absolute difference', float(account.net_r_replay_difference.abs().max()), flush=True)
strategy['taken_in_account'] = strategy.trade_id.isin(account.trade_id)
for d in (strategy, account):
    d['side'] = np.where(d.direction=='bull','BUY','SELL')
    for field in ('fill_time','exit_time'):
        d[field+'_ny'] = d[field].dt.tz_convert('America/New_York').astype(str)
        d[field+'_ist'] = d[field].dt.tz_convert('Asia/Kolkata').astype(str)
    d['holding_minutes'] = (d.exit_time-d.fill_time).dt.total_seconds()/60

START = pd.Timestamp('2019-01-01')
END = pd.Timestamp('2026-09-30')
def key_of(t, grain):
    dt = t.dt.tz_convert('UTC').dt.tz_localize(None)
    if grain == 'year': return dt.dt.strftime('%Y')
    if grain == 'month': return dt.dt.strftime('%Y-%m')
    if grain == 'day': return dt.dt.strftime('%Y-%m-%d')
    if grain == 'broker_day':
        dt = t.dt.tz_convert('America/New_York').dt.tz_localize(None) + pd.Timedelta(hours=7)
        return dt.dt.strftime('%Y-%m-%d')
    iso = dt.dt.isocalendar()
    return iso.year.astype(str)+'-W'+iso.week.astype(str).str.zfill(2)

def periods(grain):
    days = pd.Series(pd.date_range(START, END, tz='UTC'))
    if grain == 'broker_day': return days.dt.strftime('%Y-%m-%d').tolist()
    return list(dict.fromkeys(key_of(days, grain)))

def stats(values):
    x = np.asarray(values, dtype=float)
    wins, losses, flat = int((x>0).sum()), int((x<0).sum()), int((x==0).sum())
    gp, gl = float(x[x>0].sum()), float(-x[x<0].sum())
    return dict(wins=wins, losses=losses, breakeven=flat, win_pct=100*wins/len(x) if len(x) else None,
                profit_factor=gp/gl if gl else None, net=float(x.sum()), avg=float(x.mean()) if len(x) else None,
                best=float(x.max()) if len(x) else None, worst=float(x.min()) if len(x) else None)

frames = {}
for grain in ('year','month','week','day','broker_day'):
    sk = key_of(strategy.fill_time, grain)
    ak = key_of(account.fill_time, grain)
    ck = key_of(account.exit_time, grain)
    balance = stored['start_equity']
    rows=[]
    for period in periods(grain):
        s = strategy[sk==period]; opened=account[ak==period]; closed=account[ck==period]
        ss=stats(s.net_r); cs=stats(closed.pnl)
        end=balance+cs['net']
        row=dict(period=period, strategy_trades=len(s), strategy_buy=int((s.side=='BUY').sum()), strategy_sell=int((s.side=='SELL').sum()),
                 strategy_wins=ss['wins'], strategy_losses=ss['losses'], strategy_breakeven=ss['breakeven'], strategy_win_pct=ss['win_pct'],
                 strategy_net_r=ss['net'], strategy_avg_r=ss['avg'], strategy_pf=ss['profit_factor'], strategy_best_r=ss['best'], strategy_worst_r=ss['worst'],
                 account_opened=len(opened), account_closed=len(closed), account_wins=cs['wins'], account_losses=cs['losses'], account_breakeven=cs['breakeven'],
                 account_win_pct=cs['win_pct'], account_pf=cs['profit_factor'], start_balance_usd=balance, net_pnl_usd=cs['net'], end_balance_usd=end,
                 account_return_pct=100*cs['net']/balance, commission_usd=float(closed.commission_usd.sum()), slippage_usd=float(closed.slippage_usd.sum()))
        if grain == 'week':
            y,w = period.split('-W')
            monday = pd.Timestamp.fromisocalendar(int(y),int(w),1)
            row['period_start'] = str(max(monday,START).date())
            row['period_end'] = str(min(monday+pd.Timedelta(days=6),END).date())
        for sym in ('EURUSD','GBPUSD','USDJPY'):
            row[sym+'_strategy_trades']=int((s.symbol==sym).sum())
            row[sym+'_net_r']=float(s.loc[s.symbol==sym,'net_r'].sum())
            row[sym+'_account_opened']=int((opened.symbol==sym).sum())
            row[sym+'_pnl_usd']=float(closed.loc[closed.symbol==sym,'pnl'].sum())
        rows.append(row); balance=end
    f=pd.DataFrame(rows)
    assert f.strategy_trades.sum()==7646
    assert f.account_opened.sum()==stored['trades'] and f.account_closed.sum()==stored['trades']
    assert abs(f.strategy_net_r.sum()-strategy.net_r.sum())<1e-7
    assert abs(f.net_pnl_usd.sum()-(stored['final_equity']-stored['start_equity']))<0.011
    assert abs(np.prod(1+f.account_return_pct/100)*stored['start_equity']-stored['final_equity'])<0.011
    frames[grain]=f
    f.to_csv(OUT/f'{grain}_stats.csv',index=False,float_format='%.8f')
for name,d in [(f'strategy_{len(strategy)}_trades',strategy),(f'account_{len(account)}_trades',account)]:
    d.to_csv(OUT/f'{name}.csv',index=False,float_format='%.10f')
yr=frames['year']
assert {r.period:round(r.account_return_pct,1) for r in yr.itertuples()}==stored['yearly_return_pct']
for p in source_paths:
    assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))]
summary = dict(strategy_count=len(strategy), strategy=stats(strategy.net_r), account=sim,
               monthly_positive=int((frames['month'].account_return_pct>0).sum()),monthly_negative=int((frames['month'].account_return_pct<0).sum()),
               monthly_total=len(frames['month']), source_hashes=hashes,
               strategy_active_days=int((frames['day'].strategy_trades>0).sum()),account_active_days=int((frames['day'].account_opened>0).sum()))
weekday_rows=[]
for day in ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'):
    s=strategy[strategy.fill_time.dt.day_name()==day]
    a=account[account.fill_time.dt.day_name()==day]
    ss,cs=stats(s.net_r),stats(a.pnl)
    weekday_rows.append(dict(period=day,strategy_trades=len(s),strategy_net_r=ss['net'],strategy_win_pct=ss['win_pct'],
                             account_opened=len(a),account_win_pct=cs['win_pct'],net_pnl_usd=cs['net']))
frames['weekday']=pd.DataFrame(weekday_rows)
frames['weekday'].to_csv(OUT/'weekday_stats.csv',index=False,float_format='%.8f')
def records(f): return json.loads(f.to_json(orient='records'))
payload=dict(summary=summary, tables={k:records(v) for k,v in frames.items()})
(OUT/'stats.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
print(yr[['period','strategy_trades','strategy_net_r','strategy_win_pct','account_opened','net_pnl_usd','account_return_pct','end_balance_usd']].to_string(index=False))
print(f'Reconciled: {len(strategy):,} strategy trades; {len(account):,} account trades; yearly returns, costs and ending balance match.',flush=True)
