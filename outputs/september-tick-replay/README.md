# September tick comparison — research only

All variants: fresh $10,000, approved combo risk rules, corrected September news blocks.

| Scenario | Trades | Profit USD | Return | PF | Win | Closed DD |
|---|---:|---:|---:|---:|---:|---:|
| Old M5 + corrected news | 68 | 153.21 | 1.53% | 1.1 | 54.4% | 5.3% |
| Post-entry M5 | 67 | -452.70 | -4.53% | 0.63 | 40.3% | 5.8% |
| Tick SL/TP + M5 trail | 70 | -421.43 | -4.21% | 0.67 | 41.4% | 5.5% |
| Tick SL/TP + tick trail | 70 | -443.71 | -4.44% | 0.67 | 41.4% | 5.8% |
| 1s SL/TP + M5 trail | 68 | -442.91 | -4.43% | 0.66 | 41.2% | 5.7% |
| 1s SL/TP + 1s trail | 68 | -439.01 | -4.39% | 0.69 | 42.6% | 5.7% |

Original M5 is also news-corrected for a fair timing comparison. This table is September only, not the existing multi-year account. Post-entry M5 isolates removal of movement before entry; subsequent tick variants add event ordering/actual quotes/trailing timing.

## Common closed signals before account constraints

```json
{
  "original_m5": {
    "signals": 69,
    "closed": 69,
    "net_r": 12.40681575018334,
    "pf": 1.334321615033609,
    "win_pct": 52.17391304347826
  },
  "post_entry_m5": {
    "signals": 69,
    "closed": 69,
    "net_r": -16.82157744932974,
    "pf": 0.6453854660261277,
    "win_pct": 39.130434782608695
  },
  "tick_exit_bar_trail": {
    "signals": 69,
    "closed": 69,
    "net_r": -14.786539327951546,
    "pf": 0.6807198201287562,
    "win_pct": 40.57971014492754
  },
  "tick_exit_tick_trail": {
    "signals": 69,
    "closed": 69,
    "net_r": -16.57061972756212,
    "pf": 0.6350859633643058,
    "win_pct": 40.57971014492754
  },
  "one_second_bar_trail": {
    "signals": 69,
    "closed": 69,
    "net_r": -14.917269997829408,
    "pf": 0.6778970019650504,
    "win_pct": 40.57971014492754
  },
  "one_second_fast_trail": {
    "signals": 69,
    "closed": 69,
    "net_r": -13.516917008658625,
    "pf": 0.7008293839085016,
    "win_pct": 42.028985507246375
  }
}
```

## Limits

- Frozen 79 selected entries/ML exit choices/VA/activity; intrabar zones and score updates are not tested.
- All account variants start September with a fresh $10,000, aggressive combo, no prior-month drawdown or positions; not a continuation of the 2019 account.
- Original selected ledger has no September carry-in positions; censored outcomes excluded rather than force-closed.
- tick_* processes every recorded quote (ideal); one_second_* observes only the latest available quote at whole UTC seconds. Actual poll phase, outages and broker latency are not modeled.
- Nominal limit/stop fills, $5 per lot roundtrip, 0.2-pip stop slippage; gap/latency/fill-quality stress still pending.
- M5 short exits retain stored spread approximation; tick short exits use actual Ask. That quote-model change is included in the comparison.
- News uses separate retrospective September provider calendar with -5/+20 entry block, not historical live-calendar snapshots.
- Drawdown is CLOSED balance only, not intratrade floating equity drawdown; no annualized-return claim from one month.
- Research is exploratory, not independent holdout evidence or approval to deploy. Production bot, models, live state and reports unchanged.
