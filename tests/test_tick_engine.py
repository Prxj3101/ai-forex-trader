import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest import tick_engine as te  # noqa: E402
from src.research.management import POLICIES, run_trade_full  # noqa: E402

POL = {p.name: p for p in POLICIES}


def run_state(exit_, t, a, b, entry=100.0, sl=99.0, bull=True):
    tr = te.init_state(dict(bull=bull, entry=entry, risk=abs(entry - sl), exit=exit_), int(t[0]))
    return te.feed_ticks(tr, t, a, b), tr


class TickEngineTests(unittest.TestCase):
    def test_tp_and_sl_by_tick_order(self):
        t = np.array([0, 1000, 2000]); b = np.array([100.0, 101.0, 98.0])
        self.assertEqual(run_state("tp1", t, b, b)[0], (1.0, "tp", 1000))
        b2 = np.array([100.0, 98.9, 101.0])
        r, why, ms = run_state("tp1", t, b2, b2)[0]
        self.assertEqual((why, ms), ("sl", 1000)); self.assertAlmostEqual(r, -1.1)   # real (worse) tick price

    def test_trail_moves_one_second_later_old_stop_active(self):
        t = np.array([0, 1000, 1500, 2600]); b = np.array([100.0, 102.0, 100.9, 100.9])
        res, _ = run_state("trail", t, b, b)
        self.assertEqual(res[1:], ("trail", 2600))      # 1500: old stop (-1R) still active; 2600: new stop (+1R)
        self.assertAlmostEqual(res[0], 0.9)

    def test_m5_fill_candle_ignores_pre_fill_high(self):
        hi, lo, cl, sp = (np.array(x, float) for x in ([102, 100.2], [99.95, 98.9], [100.1, 99], [0, 0]))
        r, why, _ = run_trade_full(POL["tp1"], True, 100.0, 1.0, hi, lo, cl, sp)
        self.assertEqual(why, "sl")                       # the 102 high was before the fill
        r_old, why_old, _ = run_trade_full(POL["tp1"], True, 100.0, 1.0, hi, lo, cl, sp, fill_bar_partial=False)
        self.assertEqual(why_old, "tp")                   # old (leaky) behaviour, kept only for audits


if __name__ == "__main__":
    unittest.main()
