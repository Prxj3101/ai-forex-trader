# September news check — research only

September ke 30 din check kiye: 448 listed events mein USD/EUR/GBP/JPY ke 30 high-impact rows (17 distinct currency/time groups). Times UTC mein convert kiye.

Frozen strategy ke 79 September signals mein 7 news-block window mein hain. Inmein 6 original combo account mein liye gaye the. Yeh sirf entry audit hai; corrected account trade count/profit abhi recompute nahi kiya.

Available partial EURUSD tick sample: 11 signals, 2 actual tick entries news window mein. Sept 4 NFP wala suspicious trailing trade bhi blocked hai.

News rule: relevant currency ke High event se 5 minute pehle se 20 minute baad tak nayi entry block, endpoints included. Existing open trades ko is rule se force-close nahi kiya.

Files: calendar_utc.csv/json = separate calendar; block_windows_utc.csv = pair-wise windows; september_entry_audit.csv = all selected September signals; available_tick_entry_audit.csv = existing tick sample; audit.json = checks and limitations.

## Limits

- Complete for the 30 September dates as displayed by ForexFactory on observation date, not an independently exhaustive catalogue of all economic events.
- Retrospective impact classification and realized event times are not a point-in-time live-calendar archive; BOJ release time was not necessarily known in advance.
- Only BLS employment/PPI/CPI schedule groups independently crosschecked; remaining rows rely on ForexFactory.
- Entry audit uses frozen M5 signal timestamps, not full account resimulation; account membership is the original combo ledger.
- Tick entry audit uses the previously frozen partial EURUSD research output; current downloading CSV may have been overwritten.
- May 2025 through August 2026 historical news gap remains; no revised full-history PF/return claimed.
- Existing running bot calendar, state, models and published reports are unchanged.

## Sources

- https://www.forexfactory.com/calendar?month=sep.2026
- https://www.bls.gov/schedule/2026/09_sched.htm
- https://www.forexfactory.com/timezone
