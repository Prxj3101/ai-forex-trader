import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from src.research.tick_compare import tick_trade, bar_trade, POL
from src.research.management import run_trade_full


class TickManagementTests(unittest.TestCase):
    def run_path(self, seconds, bids, asks=None, mode="tick", policy="trail1", bull=True):
        times=np.array(seconds,dtype=np.int64)*1000
        bids=np.array(bids,dtype=float)
        asks=np.array(asks if asks is not None else bids,dtype=float)
        return tick_trade(times,bids,asks,0,100,99 if bull else 101,bull,POL[policy],mode)

    def test_pre_entry_target_is_ignored(self):
        outcome=self.run_path([0,10,20,30],[102,100,99.5,99],policy="tp1")
        self.assertEqual((outcome.filled_ms,outcome.reason,outcome.r),(10000,"sl",-1))

    def test_tp_before_later_sl_is_winner(self):
        outcome=self.run_path([0,1,2],[100,101,99],policy="tp1")
        self.assertEqual((outcome.reason,outcome.r,outcome.exited_ms),("tp",1,1000))
        # Identical bar extremes lack ordering; existing M5 policy chooses SL first.
        result=run_trade_full(POL['tp1'],True,100,1,np.array([101]),np.array([99]),np.array([99]),np.array([0]))
        self.assertEqual(result[:2],(-1,"sl"))

    def test_same_bar_trail_can_change_result(self):
        fast=self.run_path([0,1,2,3],[100,102,101,99])
        slow=self.run_path([0,1,2,3],[100,102,101,99],mode="bar")
        self.assertEqual((fast.reason,fast.r),("trail",1))
        self.assertEqual((slow.reason,slow.r),("sl",-1))

    def test_bar_trail_activates_at_next_bar(self):
        outcome=self.run_path([0,10,300],[100,102,101],mode="bar")
        self.assertEqual((outcome.reason,outcome.r,outcome.exited_ms),("trail",1,300000))

    def test_buy_entry_uses_ask(self):
        outcome=self.run_path([0,1],[99.9,99.9],[100.1,100],policy="tp1")
        self.assertEqual(outcome.filled_ms,1000)
        self.assertEqual(outcome.status,"unresolved_at_data_end")

    def test_sell_exit_uses_ask(self):
        outcome=self.run_path([0,1,2],[100,99,98.9],[100.1,99.2,99],policy="tp1",bull=False)
        self.assertEqual((outcome.reason,outcome.exited_ms),("tp",2000))

    def test_no_touch_is_no_fill(self):
        self.assertEqual(self.run_path([0,1],[101,102]).status,"no_tick_fill")

    def test_fixed_entry_survives_touch_missed_by_poll(self):
        times=np.array([1000,2000,3000])
        bids=np.array([102.,100.5,99.])
        outcome=tick_trade(times,bids,bids,0,100,99,True,POL['tp1'],
                           fixed_fill_ms=1500,fill_quote=(100,100))
        self.assertEqual((outcome.filled_ms,outcome.reason,outcome.exited_ms),(1500,'sl',3000))

    def test_fixed_entry_does_not_use_earlier_poll_target(self):
        times=np.array([1000,2000,3000])
        bids=np.array([102.,100.5,99.])
        outcome=tick_trade(times,bids,bids,0,100,99,True,POL['trail1'],
                           fixed_fill_ms=1500,fill_quote=(100,100))
        self.assertEqual((outcome.reason,outcome.r),('sl',-1))

    def test_fixed_entry_spread_can_stop_at_fill(self):
        times=np.array([2000]); bids=np.array([102.])
        outcome=tick_trade(times,bids,bids,0,100,99,True,POL['tp1'],
                           fixed_fill_ms=1500,fill_quote=(98.9,100))
        self.assertEqual((outcome.reason,outcome.exited_ms),('sl',1500))

    def test_fixed_entry_requires_quote(self):
        with self.assertRaises(ValueError):
            tick_trade(np.array([1000]),np.array([100.]),np.array([100.]),
                       0,100,99,True,POL['tp1'],fixed_fill_ms=500)

    def test_short_file_does_not_force_timeout(self):
        self.assertEqual(self.run_path([0,1],[100,100.5]).status,"unresolved_at_data_end")

    def test_hold_deadline(self):
        outcome=self.run_path([0,3*86400],[100,100.5],policy="tp1")
        self.assertEqual((outcome.reason,outcome.r),("timeout",0.5))

    def test_explicit_market_bar_deadline_spans_weekend(self):
        times=np.array([0,3*86400000,4*86400000],dtype=np.int64)
        bids=np.array([100,100.5,101.0])
        outcome=tick_trade(times,bids,bids,0,100,99,True,POL['tp1'],
                           hold_deadline_ms=5*86400000)
        self.assertEqual((outcome.reason,outcome.exited_ms),('tp',4*86400000))

    def test_post_entry_bar_removes_pre_entry_high(self):
        start=pd.Timestamp('2026-09-01',tz='UTC')
        signal=SimpleNamespace(fill_time=start,entry=100,sl=99,direction='bull',exit_ml='tp1')
        bars=pd.DataFrame({'high':[102,100],'low':[100,99],'close':[100.2,99],'spread':[0,0]},
                          index=pd.DatetimeIndex([start,start+pd.Timedelta(minutes=5)]))
        base=int(start.timestamp()*1000)
        times=base+np.array([0,10000,20000,300000,400000])
        bids=np.array([102,100,100.2,100,99])
        cutoff=start+pd.Timedelta(minutes=10)
        old=bar_trade(bars,signal,cutoff)
        corrected=bar_trade(bars,signal,cutoff,(times,bids,bids,base+10000))
        self.assertEqual((old.reason,old.r),('tp',1))
        self.assertEqual((corrected.reason,corrected.r),('sl',-1))


if __name__ == "__main__":
    unittest.main()
