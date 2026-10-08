# Account-rule audit of changed tick trades

## Simple findings

We checked whether the five signals whose R outcome changed were actually selected by the existing historical combo-account simulation. All five were included. They are simulated historical trades, not real broker trades executed on the VPS.

The research replay includes all three pairs over the original history, starts with the original $10,000, and uses the shared combo risk, commission, daily limits and position limits. Its unchanged baseline exactly reproduces 6,426 trades and the stored $591,359.80 final balance. This verifies that the audit is attached to the existing account rules.

### September outlier: missing historical news coverage

The Sept 4, 2026 12:30 UTC EURUSD trade S07579 was included in the saved account simulation. Its original gross result was +5.539R; using prices after the actual tick entry gives +0.653R. That outcome correction is real within the available tick reconstruction.

However, BLS schedules Employment Situation/NFP at 08:30 Eastern on Sept 4, 2026 (12:30 UTC). The strategy's existing -5/+20 minute USD news block would exclude the entry at 12:30:07 UTC if the calendar included this event.

Source: https://www.bls.gov/schedule/2026/09_sched.htm

The local historical news calendar ends April 4, 2025. The saved historical simulation therefore did not block this September 2026 trade. We added only this officially verified event IN RESEARCH MEMORY and confirmed that it gets blocked. We did not update the production calendar or claim that the rest of September's news coverage is complete.

Therefore: this trade matters to the currently saved historical report, but should be excluded under the intended news rule with correct calendar coverage. The entry-sequence issue and calendar-coverage issue are separate.

### January: corrections remain after the existing rules

These four changed trades are covered by the historical calendar, pass its news check, and are included by the original and partial tick account replays. USD-direction, same-pair, daily and combo risk checks did not eliminate them.

| Strategy trade ID | UTC signal time | Original gross R | Tick gross R |
|---|---|---:|---:|
| S00001 | Jan 2, 2019 08:30 | +1.000 | -1.000 |
| S00003 | Jan 2, 2019 13:05 | +0.205 | +0.008 |
| S00007 | Jan 3, 2019 08:20 | +1.247 | +0.486 |
| S00008 | Jan 3, 2019 14:00 | +0.202 | +0.316 |

For example, S00001 changes from a simulated +$47.58 to -$52.38 in the partial tick replay. This is an account-impacting correction, not just a dashboard timestamp change.

## What we have and have not established

- The entry-sequence issue is present in trades that survive account rules, not just discarded signals.
- Correct news coverage removes the September NFP outlier; it does not remove the January examples.
- Partial replay changes trade availability, sizing and subsequent combo decisions. Added/removed trades can amplify small early changes through compounding and risk switching.
- Full-history balances in audit.json are SENSITIVITY EXPERIMENTS: only 16 EURUSD outcomes/timestamps are changed; all other trades remain candle-based. They are not new validated tick-performance numbers and should not replace the dashboard report.
- New signals unlocked by changed timing may still use M5 outcomes. Original ML/VA/activity selections remain frozen; there is no model retraining or intrabar signal experiment.
- No actual latency/gap-fill stress yet. Tick processing is every recorded quote, not a latest-price-only one-second poll.
- Next: complete tick coverage for the same period across all three pairs, complete historical news coverage, and an account replay where all relevant entry/exit paths are processed consistently. Then compare trailing variants and costs without tuning to these examples.

## Artifacts

- `changed_trade_audit.csv`: per-trade inclusion, news checks and P&L for each replay.
- `audit.json`: baseline reconciliation, known-event source, sensitivity replay statistics and limitations.
- `*_account.csv`: research-only replay ledgers.

Production bot, models, news calendar and published performance reports are unchanged.
