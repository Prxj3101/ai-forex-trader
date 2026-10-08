"""Read-only tick quality check; write research audit artifacts only.

September JForex display/export clock is UTC+3 (header calls it EET).
Check that interpretation against stored UTC M5 bars; never infer a fixed
offset for other months from this September-only audit.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp('2026-09-01', tz='UTC')
END = pd.Timestamp('2026-10-01', tz='UTC')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    out = ROOT / 'outputs/jforex-september-quality'
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        name = f'{symbol}_Ticks_2026.09.01_2026.10.01'
        path = ROOT / 'data/ticks/jforex' / (name + '.csv')
        d = pd.read_csv(path)
        if list(d.columns) != ['Time (EET)', 'Ask', 'Bid', 'AskVolume', 'BidVolume']:
            raise ValueError(f'Unexpected columns in {path.name}')
        local = pd.to_datetime(d['Time (EET)'], format='%Y.%m.%d %H:%M:%S.%f')
        t = local.dt.tz_localize('UTC') - pd.Timedelta(hours=3)
        invalid = ~np.isfinite(d.Ask) | ~np.isfinite(d.Bid) | (d.Bid <= 0) | (d.Ask < d.Bid)
        frame = pd.DataFrame({'bid': d.Bid.to_numpy()}, index=pd.DatetimeIndex(t))
        bars = frame.bid.resample('5min').ohlc().dropna()
        m5 = pd.read_parquet(ROOT / f'data/processed/{symbol}_M5.parquet')
        m5 = m5.loc[(m5.index >= START) & (m5.index < END)]
        common = m5.index.intersection(bars.index)
        pip = .01 if symbol == 'USDJPY' else .0001
        diff = (bars.loc[common, ['open','high','low','close']] - m5.loc[common, ['open','high','low','close']]).abs() / pip
        discrepancy = diff[(diff > 1e-7).any(axis=1)].copy()
        discrepancy.to_csv(out / f'{symbol}_ohlc_differences_pips.csv')
        original_hash = sha256(path)
        copies = [{'file': p.name, 'identical_to_original': sha256(p) == original_hash}
                  for p in sorted(path.parent.glob(name + '_(*).csv'))]
        results[symbol] = {
            'file': str(path.relative_to(ROOT)), 'sha256': original_hash,
            'ticks': len(d), 'first_utc': str(t.iloc[0]), 'last_utc': str(t.iloc[-1]),
            'milliseconds_preserved': bool((local.dt.microsecond != 0).any()),
            'chronological': bool(t.is_monotonic_increasing),
            'invalid_bid_ask_rows': int(invalid.sum()),
            'outside_requested_UTC_month': int(((t < START) | (t >= END)).sum()),
            'expected_m5_bars': len(m5), 'missing_expected_m5_bars': len(m5.index.difference(bars.index)),
            'extra_tick_m5_bars': len(bars.index.difference(m5.index)),
            'bars_with_any_ohlc_difference': len(discrepancy),
            'max_ohlc_difference_pips': diff.max().to_dict(),
            'duplicate_exports': copies,
        }
    report = {'period': 'September 2026 UTC', 'research_only': True,
              'export_clock_interpretation': 'UTC+3 for September; compared against stored UTC M5 OHLC',
              'pairs': results,
              'limits': ['Expected-bar presence does not prove every tick is present or exclude within-bar gaps.',
                         'Nonzero OHLC differences mean JForex and stored M5 sources are not identical.',
                         'No changes to bot, strategy, live state or published performance reports.',
                         'A full account replay still requires carry-in positions and exit tails outside September.']}
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    text = '# September JForex tick check\n\n'
    text += '| Pair | Ticks | Expected M5 bars | Missing M5 bars | Max OHLC difference (pips) |\n|---|---:|---:|---:|---:|\n'
    for symbol, r in results.items():
        text += f"| {symbol} | {r['ticks']:,} | {r['expected_m5_bars']:,} | {r['missing_expected_m5_bars']} | {max(r['max_ohlc_difference_pips'].values()):.2f} |\n"
    text += '\nBid/Ask, chronological order, date bounds and millisecond precision checked. See audit.json for exact flags, hashes and duplicate export checks.\n\n'
    text += '\n'.join('- ' + x for x in report['limits']) + '\n'
    (out / 'README.md').write_text(text, encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
