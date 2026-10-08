# Tick comparison: exploratory sample

EURUSD: 2026-09-01 to 2026-09-02 UTC

45,711 ticks; 288 M5 bars reconciled. All times below are UTC.

## Selected-signal comparison

| Scenario | Closed | Win rate | Net R | PF |
|---|---:|---:|---:|---:|
| baseline_m5 | 3 | 66.66666666666666 | 0.5379 | 1.461002938524593 |
| tick_exit_bar_trail | 3 | 66.66666666666666 | 0.5379 | 1.461002938524593 |
| tick_exit_tick_trail | 3 | 66.66666666666666 | 0.5379 | 1.461002938524593 |

## Exit timing

| Signal | Tick fill | Baseline exit | Tick exit | Tick minus baseline (seconds) |
|---|---|---|---|---:|
| 2026-09-01 08:05:00+00:00 | 2026-09-01 08:08:38.850000+00:00 | 2026-09-01 08:10:00+00:00 | 2026-09-01 08:22:14.116000+00:00 | 734.116 |
| 2026-09-01 15:20:00+00:00 | 2026-09-01 15:22:52.279000+00:00 | 2026-09-01 16:20:00+00:00 | 2026-09-01 16:16:34.812000+00:00 | -205.188 |
| 2026-09-01 17:40:00+00:00 | 2026-09-01 17:43:50.846000+00:00 | 2026-09-01 18:10:00+00:00 | 2026-09-01 18:06:19.769000+00:00 | -220.231 |

The earliest signal's M5 baseline uses the entire fill candle: favorable movement before the actual tick fill can affect its exit. Timing is not simply shifted earlier by switching to ticks.

Always-trail results in summary.json/CSV are counterfactuals, not selected strategy results.

## Limits
- Exploratory one-pair sample, not account returns or an independent holdout.
- Only signals whose original fill candle is in this file; excludes prior-day open positions and other pairs.
- Original ML, VA and activity selection is frozen; no signal/model retraining.
- No news filtering applied here; historical news calendar coverage must be audited before portfolio testing.
- Nominal limit/stop fills, $5/lot commission and 0.2-pip stop slippage; gaps/latency not simulated.
- Every recorded tick is processed, not a 1-second latest-price poll.
- Open outcomes at the data end are censored, not force-closed.
- Always-trail rows are counterfactual mechanics checks, not the approved strategy.
- Initial prototype covers fixed TP and trailing only; account/risk/news integration remains pending.