"""Console-only audit of stored ML labels; never train or write artifacts.

Frozen historical rows/features/touches. These exploratory months do not
represent the whole training set or a regenerated entry/touch dataset.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.research.tick_compare import POL, tick_trade, bar_trade, net_r
from src.research.management import run_trade_full


def audit_month(start_text,end_text,offset):
    start,end=pd.Timestamp(start_text,tz='UTC'),pd.Timestamp(end_text,tz='UTC')
    rows=[]
    for symbol in ['EURUSD','GBPUSD','USDJPY']:
        d=pd.read_parquet(ROOT/f'data/research/{symbol}_v2.parquet')
        g=d[(d.fill_time>=start)&(d.fill_time<end)]
        ticks=pd.read_csv(ROOT/f'data/ticks/jforex/{symbol}_Ticks_{start:%Y.%m.%d}_{end:%Y.%m.%d}.csv')
        dt=pd.to_datetime(ticks['Time (EET)'],format='%Y.%m.%d %H:%M:%S.%f').dt.tz_localize('UTC')-pd.Timedelta(hours=offset)
        valid=(dt>=start)&(dt<end)
        times=dt[valid].array.as_unit('ms').asi8
        bids=ticks.Bid.to_numpy(float)[valid]; asks=ticks.Ask.to_numpy(float)[valid]
        assert np.all(np.diff(times)>=0) and np.all(asks>=bids)
        m5=pd.read_parquet(ROOT/f'data/processed/{symbol}_M5.parquet')
        m5=m5.astype({c:float for c in ['open','high','low','close','spread']})
        original=[m5[c].to_numpy() for c in ['high','low','close','spread']]
        baseline_mismatch=0
        for row in g.itertuples():
            f=int(m5.index.searchsorted(row.fill_time)); finish=min(f+864,len(m5))
            old={}
            # Reproduce the saved gross labels before interpreting any change.
            mismatch=[]
            for col,name in [('r_trail','trail1'),('r_tp1','tp1')]:
                rr,why,_=run_trade_full(POL[name],row.direction=='bull',row.entry,row.risk,
                                       *(a[f:finish] for a in original))
                if abs(rr-getattr(row,col))>.000501 or why!=getattr(row,col+'_exit'):
                    mismatch.append(col)
                old[col]=getattr(row,col)
            if mismatch:
                # A saved data-end timeout may be a provisional label from an
                # older shorter M5 snapshot. Do not attribute it to timing.
                assert all(getattr(row,c+'_exit')=='timeout' for c in mismatch), (symbol,row.Index,'Non-timeout baseline mismatch')
                baseline_mismatch+=1
                rows.append({'status':'stored_timeout_mismatch','symbol':symbol,'index':row.Index})
                continue
            deadline=int((m5.index[f+863]+pd.Timedelta(minutes=5)).timestamp()*1000) if f+863<len(m5) else int(end.timestamp()*1000)+1
            args=(times,bids,asks,int(row.fill_time.timestamp()*1000),row.entry,row.sl,row.direction=='bull')
            trail=tick_trade(*args,POL['trail1'],trail_mode='second',hold_deadline_ms=deadline)
            tp=tick_trade(*args,POL['tp1'],hold_deadline_ms=deadline)
            if trail.filled_ms is None:
                rows.append({'status':'no_tick_fill','symbol':symbol})
                continue
            signal=SimpleNamespace(fill_time=row.fill_time,entry=row.entry,sl=row.sl,
                                   direction=row.direction,exit_ml='trail')
            clipped=bar_trade(m5,signal,end,(times,bids,asks,trail.filled_ms))
            if any(x.status!='closed' for x in [trail,tp,clipped]):
                rows.append({'status':'censored','symbol':symbol})
                continue
            commission=5/(100000*(row.risk if symbol.endswith('USD') else row.risk/row.entry))
            slip=.2*(.01 if symbol=='USDJPY' else .0001)/row.risk
            def old_net(col):
                return old[col]-commission-(slip if getattr(row,col+'_exit') in ['sl','trail','be'] else 0)
            ot,op=old_net('r_trail'),old_net('r_tp1')
            nt,np1=net_r(symbol,row.entry,row.sl,trail),net_r(symbol,row.entry,row.sl,tp)
            cp=net_r(symbol,row.entry,row.sl,clipped)
            old_edge=float(np.clip(ot-op,-3,3)); new_edge=float(np.clip(nt-np1,-3,3))
            rows.append({'status':'closed','symbol':symbol,'index':row.Index,
                'fill_time':str(row.fill_time),'old_win':ot>0,'clipped_win':cp>0,'new_win':nt>0,
                'old_net_trail':ot,'new_net_trail':nt,'old_edge':old_edge,'new_edge':new_edge,
                'old_trail_better':old_edge>0,'new_trail_better':new_edge>0})
        print(f'{start:%Y-%m} {symbol}: {len(g)} stored ML rows checked, baseline mismatches {baseline_mismatch}',flush=True)
    frame=pd.DataFrame(rows)
    closed=frame[frame.status=='closed']
    ow=closed.old_win.astype(bool); nw=closed.new_win.astype(bool)
    examples=closed[ow!=nw].head(3)[['symbol','fill_time','old_net_trail','new_net_trail']].to_dict('records')
    result={'period':f'{start:%Y-%m}','stored_rows':len(frame),'compared_closed_rows':len(closed),
        'no_tick_fill':int((frame.status=='no_tick_fill').sum()),'censored':int((frame.status=='censored').sum()),
        'stored_timeout_mismatch_excluded':int((frame.status=='stored_timeout_mismatch').sum()),
        'old_net_winners':int(ow.sum()),'corrected_net_winners':int(nw.sum()),
        'winner_to_loser':int((ow&~nw).sum()),'loser_to_winner':int((~ow&nw).sum()),
        'win_label_flips_after_only_entry_clip':int((ow!=closed.clipped_win.astype(bool)).sum()),
        'win_label_flips_server_stops_second_trail':int((ow!=nw).sum()),
        'exit_target_changed':int((abs(closed.old_edge-closed.new_edge)>1e-3).sum()),
        'which_exit_better_flipped':int((closed.old_trail_better!=closed.new_trail_better).sum()),
        'examples':examples}
    print(json.dumps(result,indent=2),flush=True)
    return result


def fingerprints():
    paths=['models/zone_brain.txt','models/exit_brain.txt','data/research/ml_scores.parquet','data/research/ml_walkforward.json']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


if __name__=='__main__':
    before=fingerprints()
    audit_month('2019-01-01','2019-02-01',2)
    audit_month('2026-09-01','2026-10-01',3)
    assert fingerprints()==before, 'Protected artifacts changed during read-only audit'
    print('Models, scores and production ML report hashes unchanged.',flush=True)
