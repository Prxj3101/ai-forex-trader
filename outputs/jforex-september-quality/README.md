# September JForex tick check

| Pair | Ticks | Expected M5 bars | Missing M5 bars | Max OHLC difference (pips) |
|---|---:|---:|---:|---:|
| EURUSD | 1,387,649 | 6,336 | 0 | 0.40 |
| GBPUSD | 1,943,266 | 6,336 | 0 | 0.60 |
| USDJPY | 3,475,957 | 6,333 | 0 | 1.20 |

Bid/Ask, chronological order, date bounds and millisecond precision checked. See audit.json for exact flags, hashes and duplicate export checks.

- Expected-bar presence does not prove every tick is present or exclude within-bar gaps.
- Nonzero OHLC differences mean JForex and stored M5 sources are not identical.
- No changes to bot, strategy, live state or published performance reports.
- A full account replay still requires carry-in positions and exit tails outside September.
