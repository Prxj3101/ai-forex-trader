import unittest
import numpy as np
from src.research.tick_compare import tick_trade, POL


class ServerStopSecondTrailTests(unittest.TestCase):
    def run_path(self,times,bids,policy='trail1',bull=True):
        t=np.array(times,dtype=np.int64)
        b=np.array(bids,dtype=float)
        return tick_trade(t,b,b,0,100,99 if bull else 101,bull,POL[policy],trail_mode='second')

    def test_initial_sl_between_polls(self):
        x=self.run_path([0,250,1000],[100,99,100])
        self.assertEqual((x.reason,x.exited_ms),('sl',250))

    def test_tp_between_polls(self):
        x=self.run_path([0,250,1000],[100,101,100],policy='tp1')
        self.assertEqual((x.reason,x.exited_ms),('tp',250))

    def test_unobserved_spike_does_not_move_trail(self):
        x=self.run_path([0,100,200,1000,1500],[100,102,100.5,100.5,99])
        self.assertEqual((x.reason,x.r,x.exited_ms),('sl',-1,1500))

    def test_uploaded_trail_hits_between_updates(self):
        x=self.run_path([0,900,1000,1250,2000],[100,102,102,101,102])
        self.assertEqual((x.reason,x.r,x.exited_ms),('trail',1,1250))

    def test_poll_uses_latest_prior_quote(self):
        x=self.run_path([0,900,1001,1500],[100,100.5,102,99])
        self.assertEqual((x.reason,x.r,x.exited_ms),('sl',-1,1500))

    def test_short_uploaded_trail(self):
        x=self.run_path([0,900,1000,1250],[100,98,98,99],bull=False)
        self.assertEqual((x.reason,x.r,x.exited_ms),('trail',1,1250))


if __name__=='__main__':
    unittest.main()
