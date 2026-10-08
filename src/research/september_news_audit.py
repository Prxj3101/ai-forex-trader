"""Build a separately observed September calendar and audit frozen trade entries.

Research only: never writes production calendars, models or account reports.
The observation is retrospective; impact ratings may differ from the live week.
"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config
from src.features.news import duration_for
from src.risk.news_rules import NewsGuard

OUT = ROOT / 'outputs/news-calendar-september2026'
SOURCE = 'https://www.forexfactory.com/calendar?month=sep.2026'


def main():
    raw = pd.read_csv(OUT / 'observed_high_impact.csv')
    assert len(raw) == 30 and raw.ff_event_id.is_unique
    events = raw.copy()
    events['time'] = pd.to_datetime(raw.local_date + ' ' + raw.local_time).dt.tz_localize('Asia/Kolkata').dt.tz_convert('UTC')
    events['impact'] = 'High'
    events['source'] = 'ForexFactory retrospective browser observation 2026-10-06'
    events['source_url'] = SOURCE
    events['duration_min'] = events.event.map(duration_for)
    events['verification'] = 'ForexFactory displayed time and high-impact marker'
    bls = 'https://www.bls.gov/schedule/2026/09_sched.htm'
    events['official_crosscheck_url'] = ''
    events.loc[events.ff_event_id.isin([146935,146911,146923,148403,148388,148367,153056,148336,148351]), 'official_crosscheck_url'] = bls
    # Key DST and local-date rollover checks, independently matched to schedules.
    def at(eid):
        return events.loc[events.ff_event_id == eid, 'time'].iloc[0]
    assert at(146911) == pd.Timestamp('2026-09-04 12:30', tz='UTC')
    assert at(148388) == pd.Timestamp('2026-09-10 12:30', tz='UTC')
    assert at(148336) == pd.Timestamp('2026-09-11 12:30', tz='UTC')
    assert at(146519) == pd.Timestamp('2026-09-16 18:30', tz='UTC')
    assert at(150538) == pd.Timestamp('2026-09-10 12:15', tz='UTC')
    events.to_csv(OUT / 'calendar_utc.csv', index=False)
    (OUT / 'calendar_utc.json').write_text(events.to_json(orient='records', date_format='iso', indent=2), encoding='utf-8')
    guard = NewsGuard(events, config.profile(), block=True)
    assert guard.block_on and not guard.on, 'This audit requires the current evaluation profile'
    # Verify inclusive window endpoints and currency isolation.
    for t in ['12:25', '12:30', '12:50']:
        assert guard.blocked('EURUSD', pd.Timestamp('2026-09-04 ' + t, tz='UTC'))
    for t in ['12:24:59', '12:50:01']:
        assert not guard.blocked('EURUSD', pd.Timestamp('2026-09-04 ' + t, tz='UTC'))
    assert not guard.blocked('USDJPY', pd.Timestamp('2026-09-04 08:50', tz='UTC'))
    windows = []
    for symbol in ['EURUSD','GBPUSD','USDJPY']:
        w = guard.block_windows(symbol).copy()
        w['symbol'] = symbol
        windows.append(w)
    pd.concat(windows).to_csv(OUT / 'block_windows_utc.csv', index=False)
    ledger = pd.read_csv(ROOT / 'outputs/trade-stats-2019-2026/strategy_7646_trades.csv')
    ledger['fill_time'] = pd.to_datetime(ledger.fill_time, utc=True)
    sept = ledger[(ledger.fill_time >= pd.Timestamp('2026-09-01',tz='UTC')) & (ledger.fill_time < pd.Timestamp('2026-10-01',tz='UTC'))].copy()
    account = pd.read_csv(ROOT / 'outputs/trade-stats-2019-2026/account_6426_trades.csv')
    sept['taken_in_original_combo'] = sept.trade_id.isin(account.trade_id)
    sept['research_news_block_reason'] = [guard.blocked(r.symbol,r.fill_time) for r in sept.itertuples()]
    sept['research_news_blocked'] = sept.research_news_block_reason.notna()
    sept.to_csv(OUT / 'september_entry_audit.csv', index=False)
    partial = pd.read_csv(ROOT / 'outputs/tick-comparison-september-partial/trade_comparison.csv')
    partial = partial[partial.scenario == 'tick_exit_tick_trail'].copy()
    partial['research_news_block_reason'] = [guard.blocked(r.symbol,pd.Timestamp(r.fill_time)) if pd.notna(r.fill_time) else None for r in partial.itertuples()]
    partial.to_csv(OUT / 'available_tick_entry_audit.csv', index=False)
    summary = {
        'research_only': True, 'observation_date': '2026-10-06',
        'source_url': SOURCE, 'display_timezone': 'Asia/Kolkata (GMT+05:30), verified on /timezone',
        'observed_month_dates': 30, 'observed_all_currency_event_rows': 448,
        'unpopulated_observed_event_rows': 0,
        'relevant_high_impact_rows': len(events),
        'distinct_currency_event_times': len(events[['currency','time']].drop_duplicates()),
        'counts_by_currency': events.currency.value_counts().to_dict(),
        'september_frozen_selected_signals': len(sept),
        'selected_signals_in_news_windows': int(sept.research_news_blocked.sum()),
        'original_combo_trades_in_news_windows': int((sept.research_news_blocked & sept.taken_in_original_combo).sum()),
        'available_tick_signals': len(partial),
        'available_tick_entries_blocked': int(partial.research_news_block_reason.notna().sum()),
        'limits': [
            'Complete for the 30 September dates as displayed by ForexFactory on observation date, not an independently exhaustive catalogue of all economic events.',
            'Retrospective impact classification and realized event times are not a point-in-time live-calendar archive; BOJ release time was not necessarily known in advance.',
            'Only BLS employment/PPI/CPI schedule groups independently crosschecked; remaining rows rely on ForexFactory.',
            'Entry audit uses frozen M5 signal timestamps, not full account resimulation; account membership is the original combo ledger.',
            'Tick entry audit uses the previously frozen partial EURUSD research output; current downloading CSV may have been overwritten.',
            'May 2025 through August 2026 historical news gap remains; no revised full-history PF/return claimed.',
            'Existing running bot calendar, state, models and published reports are unchanged.'
        ]
    }
    (OUT / 'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text = '# September news check — research only\n\n'
    text += f"September ke 30 din check kiye: 448 listed events mein USD/EUR/GBP/JPY ke {len(events)} high-impact rows ({summary['distinct_currency_event_times']} distinct currency/time groups). Times UTC mein convert kiye.\n\n"
    text += f"Frozen strategy ke {len(sept)} September signals mein {summary['selected_signals_in_news_windows']} news-block window mein hain. Inmein {summary['original_combo_trades_in_news_windows']} original combo account mein liye gaye the. Yeh sirf entry audit hai; corrected account trade count/profit abhi recompute nahi kiya.\n\n"
    text += f"Available partial EURUSD tick sample: {len(partial)} signals, {summary['available_tick_entries_blocked']} actual tick entries news window mein. Sept 4 NFP wala suspicious trailing trade bhi blocked hai.\n\n"
    text += 'News rule: relevant currency ke High event se 5 minute pehle se 20 minute baad tak nayi entry block, endpoints included. Existing open trades ko is rule se force-close nahi kiya.\n\n'
    text += 'Files: calendar_utc.csv/json = separate calendar; block_windows_utc.csv = pair-wise windows; september_entry_audit.csv = all selected September signals; available_tick_entry_audit.csv = existing tick sample; audit.json = checks and limitations.\n\n'
    text += '## Limits\n\n' + '\n'.join('- '+x for x in summary['limits'])
    text += '\n\n## Sources\n\n- '+SOURCE+'\n- '+bls+'\n- https://www.forexfactory.com/timezone\n'
    (OUT / 'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
