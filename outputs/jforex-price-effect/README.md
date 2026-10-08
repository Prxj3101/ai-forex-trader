# September price differences: isolated check

Checked 79 frozen selected trades: 79 September entries and 0 carry-in positions. Old management outcomes reconcile to the stored selected ledger.

Selected trade results/exit bars changed: 0. Entry touch checks changed: 0. Gross R difference: 0.00000000R.

Positions encountering a different candle before exit: 1. End-censored selected positions: 0.

Only price-source differences are isolated here. Faster SL/TP/trailing requires the separate tick replay.

- Frozen entries, zones, ML scores and selected exit policies; signal generation/retraining is not tested.
- Only BID OHLC discrepancies are changed; spread/cost/risk settings and M5 timing stay fixed.
- This does not test seconds-based execution, pre-entry contamination correction, slippage or bid/ask model differences.
- Counterfactual tp1/trail1 checks are diagnostics, not a recommended exit strategy.
- End-censored positions cannot establish their final outcome; no revised full account return claimed.
- Original account membership and research news flags are annotations, not a new portfolio simulation.