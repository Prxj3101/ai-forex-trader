# Exploratory tick comparison: available partial downloads

This research does not change the running bot, models or published performance reports.

## What was available

- EURUSD, 1-14 September 2026: 579,992 ticks; all 2,880 stored M5 bars have tick OHLC. Maximum OHLC difference is 0.1 pip (high); open/low/close match exactly.
- EURUSD, 1-4 January 2019: 572,046 ticks; all 864 stored trading M5 bars have tick OHLC. Maximum difference is 0.1 pip (close).
- These are partial monthly CSV files left by interrupted downloads, not completed monthly exports. The September cutoff excludes the incomplete 15 September. No GBPUSD or USDJPY monthly file was available.

## Results on identical fully resolved selected signals

Original ML, value-area and activity selections are frozen. Figures are net R after $5/lot commission and 0.2-pip stop slippage. These are signal-level comparisons BEFORE historical news and portfolio limits, not account returns.

| Period | Variant | Signals | Win rate | Net R | PF |
|---|---|---:|---:|---:|---:|
| Sep 1-14 2026 | Original M5 | 11 | 36.4% | +0.044 | 1.006 |
| Sep 1-14 2026 | M5 exits, exclude pre-entry price | 11 | 36.4% | -4.842 | 0.393 |
| Sep 1-14 2026 | Tick exits, trail moves at M5 close | 11 | 36.4% | -4.842 | 0.393 |
| Sep 1-14 2026 | Tick exits, trail moves on ticks | 11 | 36.4% | -4.842 | 0.393 |
| Jan 1-4 2019 | Original M5 | 5 | 80.0% | +0.908 | 1.798 |
| Jan 1-4 2019 | M5 exits, exclude pre-entry price | 5 | 20.0% | -3.438 | 0.033 |
| Jan 1-4 2019 | Tick exits, trail moves at M5 close | 5 | 20.0% | -3.438 | 0.033 |
| Jan 1-4 2019 | Tick exits, trail moves on ticks | 5 | 40.0% | -1.952 | 0.156 |

## Interpretation

The key difference in September comes from removing movement before the trade actually filled, rather than from speeding up the trailing stop. One trailing signal at 12:30 UTC on September 4 falls from +5.539 gross R (+5.411 net R) in the original M5 calculation to +0.653 gross R (+0.526 net R). The baseline used the fill candle's pre-entry high to ratchet its trailing stop. The other ten signals retain their gross R outcome, although exit timestamps change.

January demonstrates that intrabar trailing can itself change outcomes: tick trailing improves the corrected bar-trailing sample, but does not restore the original M5 result. Four of five gross R outcomes change when pre-entry movement is excluded. This does not establish which trailing method is best over the full strategy history.

## Practical limits and pending work

- Tiny, previously researched samples from one pair; no independent holdout and no tuning to their winners.
- Signals include no historical news filtering in this experiment. A news-time outlier may be excluded by the actual account rules; do not extrapolate these numbers to the approved combo account.
- Simultaneous signals are evaluated independently; max-one-per-pair, daily limits, combo risk, prior-day open trades and cross-pair contention are not simulated here.
- Nominal limit/stop fills; actual gap fills and latency need separate stress tests. Processing all recorded ticks differs from polling only the latest quote once per second.
- M5 reconciliation supports price consistency, but cannot prove every historical tick was captured. Large weekend gaps are not automatically data failures.
- Full-month/all-pair testing and news/portfolio integration remain pending. Keep the live bot unchanged while this research is reviewed.

Per-signal CSV and audit JSON are in `tick-comparison-september-partial` and `tick-comparison-january2019-partial` alongside this file.
