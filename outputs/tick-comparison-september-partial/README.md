# Tick comparison: exploratory sample

EURUSD: 2026-09-01 to 2026-09-15 UTC

579,992 ticks; 2880 M5 bars reconciled. All times below are UTC.

## Selected-signal comparison

| Scenario | Closed | Win rate | Net R | PF |
|---|---:|---:|---:|---:|
| baseline_m5 | 11 | 36.36363636363637 | 0.0441 | 1.0055319378492735 |
| post_entry_m5_exit | 11 | 36.36363636363637 | -4.8417 | 0.39282244895303864 |
| tick_exit_bar_trail | 11 | 36.36363636363637 | -4.8417 | 0.39282244895303864 |
| tick_exit_tick_trail | 11 | 36.36363636363637 | -4.8417 | 0.39282244895303864 |

## Exit timing

| Signal | Tick fill | Baseline exit | Tick exit | Tick minus baseline (seconds) |
|---|---|---|---|---:|
| 2026-09-01 08:05:00+00:00 | 2026-09-01 08:08:38.850000+00:00 | 2026-09-01 08:10:00+00:00 | 2026-09-01 08:22:14.116000+00:00 | 734.116 |
| 2026-09-01 15:20:00+00:00 | 2026-09-01 15:22:52.279000+00:00 | 2026-09-01 16:20:00+00:00 | 2026-09-01 16:16:34.812000+00:00 | -205.188 |
| 2026-09-01 17:40:00+00:00 | 2026-09-01 17:43:50.846000+00:00 | 2026-09-01 18:10:00+00:00 | 2026-09-01 18:06:19.769000+00:00 | -220.231 |
| 2026-09-02 08:00:00+00:00 | 2026-09-02 08:02:20.539000+00:00 | 2026-09-02 09:00:00+00:00 | 2026-09-02 08:57:27.406000+00:00 | -152.594 |
| 2026-09-03 15:05:00+00:00 | 2026-09-03 15:06:05.410000+00:00 | 2026-09-03 16:20:00+00:00 | 2026-09-03 16:16:10.679000+00:00 | -229.321 |
| 2026-09-04 12:30:00+00:00 | 2026-09-04 12:30:01.382000+00:00 | 2026-09-04 12:35:00+00:00 | 2026-09-04 12:30:01.595000+00:00 | -298.405 |
| 2026-09-04 12:30:00+00:00 | 2026-09-04 12:30:07.560000+00:00 | 2026-09-04 12:40:00+00:00 | 2026-09-04 12:30:40.444000+00:00 | -559.556 |
| 2026-09-09 11:20:00+00:00 | 2026-09-09 11:22:10.828000+00:00 | 2026-09-09 11:30:00+00:00 | 2026-09-09 11:27:39.291000+00:00 | -140.709 |
| 2026-09-09 15:05:00+00:00 | 2026-09-09 15:07:06.984000+00:00 | 2026-09-09 15:20:00+00:00 | 2026-09-09 15:18:51.198000+00:00 | -68.802 |
| 2026-09-14 15:35:00+00:00 | 2026-09-14 15:37:56.202000+00:00 | 2026-09-14 16:15:00+00:00 | 2026-09-14 16:12:03.691000+00:00 | -176.309 |
| 2026-09-14 16:15:00+00:00 | 2026-09-14 16:15:46.781000+00:00 | 2026-09-14 16:45:00+00:00 | 2026-09-14 16:43:20.417000+00:00 | -99.583 |

The M5 baseline uses the entire fill candle: favorable movement before the actual tick fill can affect its exit. Timing is not simply shifted earlier by switching to ticks.

## Common fully resolved signals (same denominator)

```json
{
  "baseline_m5": {
    "matched_closed_signals": 11,
    "net_r_sum": 0.04411193614505704,
    "win_pct": 36.36363636363637,
    "pf": 1.0055319378492735
  },
  "post_entry_m5_exit": {
    "matched_closed_signals": 11,
    "net_r_sum": -4.841662739217783,
    "win_pct": 36.36363636363637,
    "pf": 0.39282244895303864
  },
  "tick_exit_bar_trail": {
    "matched_closed_signals": 11,
    "net_r_sum": -4.841662739217783,
    "win_pct": 36.36363636363637,
    "pf": 0.39282244895303864
  },
  "tick_exit_tick_trail": {
    "matched_closed_signals": 11,
    "net_r_sum": -4.841662739217783,
    "win_pct": 36.36363636363637,
    "pf": 0.39282244895303864
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