"""Inventory September tick download coverage; no downloads or live mutations."""
from pathlib import Path
import json
from urllib.parse import unquote
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
START=pd.Timestamp('2026-09-01',tz='UTC')
END=pd.Timestamp('2026-10-01',tz='UTC')


def main():
    out=ROOT/'outputs/tick-data-plan'
    out.mkdir(parents=True,exist_ok=True)
    cache={}
    for path in (ROOT/'data/ticks/cache').glob('*.json'):
        raw=json.loads(path.read_text())
        symbol=unquote(path.name).split('/')[1].replace('-','')
        timestamp=pd.to_datetime(raw['timestamp'],unit='ms',utc=True).floor('h')
        cache[(symbol,timestamp)]=path.name
    csv_hours={}
    csv_files=[]
    for path in (ROOT/'data/ticks').rglob('*.csv'):
        symbol=path.name.split('-')[0].upper()
        if symbol not in ('EURUSD','GBPUSD','USDJPY'):
            continue
        stamps=pd.read_csv(path,usecols=['timestamp']).timestamp
        timestamps=pd.to_datetime(stamps,unit='ms',utc=True)
        timestamps=timestamps[(timestamps>=START)&(timestamps<END)]
        if timestamps.empty:
            continue
        csv_files.append({'file':str(path.relative_to(ROOT)), 'first':str(timestamps.min()),'last':str(timestamps.max())})
        for hour in timestamps.dt.floor('h').unique():
            csv_hours.setdefault((symbol,hour),[]).append(str(path.relative_to(ROOT)))
    rows=[]
    summaries={}
    for symbol in ('EURUSD','GBPUSD','USDJPY'):
        m5=pd.read_parquet(ROOT/f'data/processed/{symbol}_M5.parquet',columns=['close'])
        hours=m5.loc[(m5.index>=START)&(m5.index<END)].index.floor('h').unique().sort_values()
        for hour in hours:
            rows.append({'symbol':symbol,'hour_utc':str(hour),'csv_has_ticks':(symbol,hour) in csv_hours,
                         'cache_present':(symbol,hour) in cache,
                         'download_needed':(symbol,hour) not in csv_hours and (symbol,hour) not in cache})
        group=[r for r in rows if r['symbol']==symbol]
        summaries[symbol]={'stored_market_hours':len(group),'csv_hours':sum(r['csv_has_ticks'] for r in group),
                           'cached_hours':sum(r['cache_present'] for r in group),
                           'hours_without_csv_or_cache':sum(r['download_needed'] for r in group)}
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'september_hour_inventory.csv',index=False)
    frame[frame.download_needed].to_csv(out/'september_missing_hours.csv',index=False)
    result={'start':str(START),'end':str(END),'pairs':summaries,'csv_sources':csv_files,
            'limits':['Presence is not proof of complete tick coverage; final CSVs still need OHLC/gap verification.',
                      'Expected hours come from stored M5 candles; source gaps outside those candles are not detectable here.',
                      'Full account replay also needs carry-in trades/positions from before September and any later exit tails.',
                      'This inventory does not contact providers or bypass rate limits.']}
    (out/'coverage.json').write_text(json.dumps(result,indent=2))
    text=['# September tick-data plan','', 'Keep live bot unchanged. Complete this fixed period for all three pairs before interpreting portfolio results.',
          '', '| Pair | Stored market hours | Hours with CSV ticks | Hours cached | Hours with neither |', '|---|---:|---:|---:|---:|']
    for symbol,s in summaries.items():
        text.append(f"| {symbol} | {s['stored_market_hours']} | {s['csv_hours']} | {s['cached_hours']} | {s['hours_without_csv_or_cache']} |")
    text.extend(['','CSV and cache counts overlap; do not add them. Existing partial EURUSD CSVs are preserved.',
                 '', '## Next work', '', '- Complete/resume September exports using provider-supported access, reusing cache.',
                 '- Complete September high-impact news coverage; the existing historical calendar ends April 2025. One NFP event alone is insufficient.',
                 '- Verify tick data, include carry-in positions and exit tails, then compare corrected M5 trailing with tick trailing under the same account rules.',
                 '- No full-history tick-return claims from a partial-period hybrid replay.','', '## Inventory limits',''])
    text.extend('- '+s for s in result['limits'])
    (out/'README.md').write_text('\n'.join(text))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
