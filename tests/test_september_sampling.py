import unittest
import numpy as np
from src.research.september_tick_replay import sample_one_second


class SecondSamplingTests(unittest.TestCase):
    def test_observation_never_uses_future_tick(self):
        times=np.array([118,999,1001,1800,3000],dtype=np.int64)
        bids=np.array([100,101,102,103,104],dtype=float)
        t,b,a=sample_one_second(times,bids,bids+.1)
        np.testing.assert_array_equal(t,[1000,2000,3000])
        np.testing.assert_array_equal(b,[101,103,104])
        np.testing.assert_allclose(a,b+.1)

    def test_stale_quote_is_observed_without_using_future_update(self):
        times=np.array([100,5000],dtype=np.int64)
        bids=np.array([100,105],dtype=float)
        t,b,_=sample_one_second(times,bids,bids)
        np.testing.assert_array_equal(t,[1000,2000,3000,4000,5000])
        np.testing.assert_array_equal(b,[100,100,100,100,105])


if __name__=='__main__':
    unittest.main()
