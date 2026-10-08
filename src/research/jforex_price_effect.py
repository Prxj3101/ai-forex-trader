"""Isolate September exported OHLC discrepancies from execution timing changes.

Frozen signals, levels, ML choices, spread and M5 management rules. Swap only
September BID OHLC with JForex tick-derived bars, keeping source files intact.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research.management import HOLD_BARS, POLICIES, run_trade_full
from src.risk.news_rules import NewsGuard
from src import config

START = pd.Timestamp('2026-09-01', tz='UTC')
END = pd.Timestamp('2026-10-01', tz='UTC')
POL = {p.name:p for p in POLICIES}


def main():
    out = ROOT / 'outputs/jforex-price-effect'
    out.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ROOT / 'outputs/trade-stats-2019-2026/strategy_7646_trades.csv')
    ledger['fill_time'] = pd.to_datetime(ledger.fill_time, utc=True)
    ledger['exit_time'] = pd.to_datetime(ledger.exit_time, utc=True)
    # Include selected positions entered earlier that could encounter September bars.
    candidates = ledger[(ledger.fill_time < END) & (ledger.exit_time >= START)].copy()
    account = pd.read_csv(ROOT / 'outputs/trade-stats-2019-2026/account_6426_trades.csv')
    events = pd.read_csv(ROOT / 'outputs/news-calendar-september2026/calendar_utc.csv')
    events['time'] = pd.to_datetime(events.time, utc=True)
    guard = NewsGuard(events, config.profile(), block=True)
    rows = []
    counts = {}
    for symbol in ['EURUSD','GBPUSD','USDJPY']:
        m5 = pd.read_parquet(ROOT / f'data/processed/{symbol}_M5.parquet').copy()
        m5 = m5.astype({c:float for c in ['open','high','low','close','spread']})
        d = pd.read_csv(ROOT / f'data/ticks/jforex/{symbol}_Ticks_2026.09.01_2026.10.01.csv')
        t = pd.to_datetime(d['Time (EET)'],format='%Y.%m.%d %H:%M:%S.%f').dt.tz_localize('UTC') - pd.Timedelta(hours=3)
        bars = pd.Series(d.Bid.to_numpy(),index=pd.DatetimeIndex(t)).resample('5min').ohlc().dropna()
        ix = m5.index[(m5.index >= START) & (m5.index < END)]
        assert not len(ix.difference(bars.index))
        modified = m5.copy()
        modified.loc[ix,['open','high','low','close']] = bars.loc[ix,['open','high','low','close']]
        assert modified.spread.equals(m5.spread)
        dirty = ix[((modified.loc[ix,['open','high','low','close']] - m5.loc[ix,['open','high','low','close']]).abs() > 1e-10).any(axis=1)]
        for c in candidates[candidates.symbol == symbol].itertuples():
            s = int(m5.index.searchsorted(c.fill_time))
            assert m5.index[s] == c.fill_time
            e = min(s + HOLD_BARS,len(m5))
            old, new = m5.iloc[s:e], modified.iloc[s:e]
            risk = abs(c.entry-c.sl)
            bull = c.direction == 'bull'
            def outcome(frame, policy):
                return run_trade_full(POL[policy],bull,c.entry,risk,
                                      *(frame[k].to_numpy() for k in ['high','low','close','spread']))
            chosen = 'tp1' if c.exit_ml == 'tp1' else 'trail1'
            for policy in ['tp1','trail1']:
                a,b = outcome(old,policy),outcome(new,policy)
                # Legacy result reconstruction must match the frozen selected ledger.
                if policy == chosen:
                    assert abs(a[0]-c.r) < 1e-6, (c.trade_id,a[0],c.r)
                    assert a[1] == c.reason, (c.trade_id,a[1],c.reason)
                changed = abs(a[0]-b[0]) > 1e-8 or a[1:] != b[1:]
                entry_old = old.iloc[0]
                entry_new = new.iloc[0]
                # Same existing M5 entry-touch approximation, not an actual tick fill.
                hit_old = entry_old.low + entry_old.spread <= c.entry if bull else entry_old.high >= c.entry
                hit_new = entry_new.low + entry_new.spread <= c.entry if bull else entry_new.high >= c.entry
                rows.append({'trade_id':c.trade_id,'symbol':symbol,'fill_time':str(c.fill_time),
                             'carry_in':bool(c.fill_time < START),'chosen_policy':chosen,'tested_policy':policy,
                             'is_selected_policy':policy==chosen,'original_combo_taken':c.trade_id in set(account.trade_id),
                             'research_news_block':guard.blocked(symbol,c.fill_time),
                             'old_gross_r':a[0],'new_gross_r':b[0],'r_difference':b[0]-a[0],
                             'old_reason':a[1],'new_reason':b[1],
                             'old_exit_bar':str(old.index[a[2]]),'new_exit_bar':str(new.index[b[2]]),
                             'result_or_exit_changed':changed,'entry_touch_changed':bool(hit_old != hit_new),
                             'different_bars_encountered_before_exit':int(((dirty>=c.fill_time)&(dirty<=max(old.index[a[2]],new.index[b[2]]))).sum()),
                             'end_censored': bool((a[1]=='timeout' or b[1]=='timeout') and len(old)<HOLD_BARS)})
        counts[symbol] = len(dirty)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'trade_audit.csv',index=False)
    selected = frame[frame.is_selected_policy]
    changed = selected[selected.result_or_exit_changed | selected.entry_touch_changed]
    changed.to_csv(out / 'changed_selected_trades.csv',index=False)
    result = {
        'research_only':True,'price_only_comparison':True,'legacy_selected_results_reconcile':True,
        'selected_trades_checked':len(selected),'september_entries':int((~selected.carry_in).sum()),
        'carry_in_trades':int(selected.carry_in.sum()),'different_bars_by_pair':counts,
        'selected_outcomes_or_exit_times_changed':int(selected.result_or_exit_changed.sum()),
        'selected_entry_touch_checks_changed':int(selected.entry_touch_changed.sum()),
        'selected_positions_encountering_different_bars':int((selected.different_bars_encountered_before_exit>0).sum()),
        'selected_outcomes_censored_at_data_end':int(selected.end_censored.sum()),
        'diagnostic_both_exit_policies_changed':int(frame.result_or_exit_changed.sum()),
        'selected_gross_r_difference':float(selected.r_difference.sum()),
        'changed_selected_trades':changed.to_dict(orient='records'),
        'limits':['Frozen entries, zones, ML scores and selected exit policies; signal generation/retraining is not tested.',
                  'Only BID OHLC discrepancies are changed; spread/cost/risk settings and M5 timing stay fixed.',
                  'This does not test seconds-based execution, pre-entry contamination correction, slippage or bid/ask model differences.',
                  'Counterfactual tp1/trail1 checks are diagnostics, not a recommended exit strategy.',
                  'End-censored positions cannot establish their final outcome; no revised full account return claimed.',
                  'Original account membership and research news flags are annotations, not a new portfolio simulation.']}
    (out / 'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text = '# September price differences: isolated check\n\n'
    text += f"Checked {len(selected)} frozen selected trades: {result['september_entries']} September entries and {result['carry_in_trades']} carry-in positions. Old management outcomes reconcile to the stored selected ledger.\n\n"
    text += f"Selected trade results/exit bars changed: {result['selected_outcomes_or_exit_times_changed']}. Entry touch checks changed: {result['selected_entry_touch_checks_changed']}. Gross R difference: {result['selected_gross_r_difference']:.8f}R.\n\n"
    text += f"Positions encountering a different candle before exit: {result['selected_positions_encountering_different_bars']}. End-censored selected positions: {result['selected_outcomes_censored_at_data_end']}.\n\n"
    text += 'Only price-source differences are isolated here. Faster SL/TP/trailing requires the separate tick replay.\n\n' + '\n'.join('- '+x for x in result['limits'])
    (out / 'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
