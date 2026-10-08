# Tick comparison: exploratory sample

EURUSD: 2019-01-01 to 2019-01-05 UTC

572,046 ticks; 864 M5 bars reconciled. All times below are UTC.

## Selected-signal comparison

| Scenario | Closed | Win rate | Net R | PF |
|---|---:|---:|---:|---:|
| baseline_m5 | 5 | 80.0 | 0.9078 | 1.7980040310960594 |
| post_entry_m5_exit | 5 | 20.0 | -3.4385 | 0.032625835819666445 |
| tick_exit_bar_trail | 5 | 20.0 | -3.4385 | 0.032625835819666445 |
| tick_exit_tick_trail | 5 | 40.0 | -1.9522 | 0.15581253681306 |

## Exit timing

| Signal | Tick fill | Baseline exit | Tick exit | Tick minus baseline (seconds) |
|---|---|---|---|---:|
| 2019-01-02 08:30:00+00:00 | 2019-01-02 08:30:37.677000+00:00 | 2019-01-02 08:35:00+00:00 | 2019-01-02 09:41:07.785000+00:00 | 3967.785 |
| 2019-01-02 13:05:00+00:00 | 2019-01-02 13:09:19.303000+00:00 | 2019-01-02 13:15:00+00:00 | 2019-01-02 13:20:31.310000+00:00 | 331.31 |
| 2019-01-03 08:20:00+00:00 | 2019-01-03 08:22:51.796000+00:00 | 2019-01-03 08:30:00+00:00 | 2019-01-03 08:26:31.600000+00:00 | -208.4 |
| 2019-01-03 14:00:00+00:00 | 2019-01-03 14:02:07.837000+00:00 | 2019-01-03 14:10:00+00:00 | 2019-01-03 14:10:12.984000+00:00 | 12.984 |
| 2019-01-04 13:30:00+00:00 | 2019-01-04 13:30:01.483000+00:00 | 2019-01-04 13:35:00+00:00 | 2019-01-04 13:30:01.483000+00:00 | -298.517 |

The M5 baseline uses the entire fill candle: favorable movement before the actual tick fill can affect its exit. Timing is not simply shifted earlier by switching to ticks.

## Common fully resolved signals (same denominator)

```json
{
  "baseline_m5": {
    "matched_closed_signals": 5,
    "net_r_sum": 0.9078337534266172,
    "win_pct": 80.0,
    "pf": 1.7980040310960594
  },
  "post_entry_m5_exit": {
    "matched_closed_signals": 5,
    "net_r_sum": -3.4384962238754957,
    "win_pct": 20.0,
    "pf": 0.032625835819666445
  },
  "tick_exit_bar_trail": {
    "matched_closed_signals": 5,
    "net_r_sum": -3.4384962238754957,
    "win_pct": 20.0,
    "pf": 0.032625835819666445
  },
  "tick_exit_tick_trail": {
    "matched_closed_signals": 5,
    "net_r_sum": -1.9521890722578388,
    "win_pct": 40.0,
    "pf": 0.15581253681306
  }
}
```

Always-trail results in summary.json/CSV are counterfactuals, not selected strategy results.

## Limits
- Exploratory one-pair sample, not account returns or an independent holdout.
- Only signals whose original fill candle is in this file; excludes prior-day open positions and other pairs.
- Original ML, VA and activity selection is frozen; no signal/model retraining.
- post_entry_m5_exit clips the fill candle to post-fill BID extremes while retaining the original M5 spread approximation.
- No news filtering applied here; historical news calendar coverage must be audited before portfolio testing.
- Nominal limit/stop fills, $5/lot commission and 0.2-pip stop slippage; gaps/latency not simulated.
- Every recorded tick is processed, not a 1-second latest-price poll.
- Open outcomes at the data end are censored, not force-closed.
- Always-trail rows are counterfactual mechanics checks, not the approved strategy.
- Initial prototype covers fixed TP and trailing only; account/risk/news integration remains pending.