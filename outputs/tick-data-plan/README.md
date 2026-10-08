# September tick-data plan

Keep live bot unchanged. Complete this fixed period for all three pairs before interpreting portfolio results.

| Pair | Stored market hours | Hours with CSV ticks | Hours cached | Hours with neither |
|---|---:|---:|---:|---:|
| EURUSD | 528 | 253 | 253 | 275 |
| GBPUSD | 528 | 0 | 0 | 528 |
| USDJPY | 528 | 0 | 0 | 528 |

CSV and cache counts overlap; do not add them. Existing partial EURUSD CSVs are preserved.

## Next work

- Complete/resume September exports using provider-supported access, reusing cache.
- Complete September high-impact news coverage; the existing historical calendar ends April 2025. One NFP event alone is insufficient.
- Verify tick data, include carry-in positions and exit tails, then compare corrected M5 trailing with tick trailing under the same account rules.
- No full-history tick-return claims from a partial-period hybrid replay.

## Inventory limits

- Presence is not proof of complete tick coverage; final CSVs still need OHLC/gap verification.
- Expected hours come from stored M5 candles; source gaps outside those candles are not detectable here.
- Full account replay also needs carry-in trades/positions from before September and any later exit tails.
- This inventory does not contact providers or bypass rate limits.