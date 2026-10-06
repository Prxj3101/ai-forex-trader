import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.risk.portfolio_sim import RiskMode, combo_config, daily_ok, trade_risk_pct
from src.live.shadow import ShadowEngine, STATE_VERSION


class ComboRiskTests(unittest.TestCase):
    def test_thresholds_and_sticky_recovery(self):
        cfg, mode = combo_config(), RiskMode()
        mode.select(cfg, 10000, 10000)
        self.assertEqual((cfg.risk_pct, cfg.risk_cap_pct), (0.5, 1.0))
        mode.select(cfg, 9700, 10000)
        self.assertFalse(mode.aggressive)
        self.assertEqual((cfg.risk_pct, cfg.daily_loss_pct, cfg.daily_risk_budget_pct), (0.25, 1.5, 2.0))
        mode.select(cfg, 9800, 10000)
        self.assertFalse(mode.aggressive)
        mode.select(cfg, 9900, 10000)
        self.assertTrue(mode.aggressive)
        self.assertEqual((cfg.daily_loss_pct, cfg.daily_risk_budget_pct, cfg.kill_dd_pct), (2.0, 2.5, 10.0))

    def test_house_money_caps_and_brake_override(self):
        cfg, mode = combo_config(), RiskMode()
        mode.select(cfg, 10000, 10000)
        self.assertEqual(trade_risk_pct(cfg, 10000, 10000, 8000), (1.0, False))
        mode.select(cfg, 9700, 10000)
        self.assertEqual(trade_risk_pct(cfg, 9700, 10000, 8000), (0.5, False))
        mode.select(cfg, 9500, 10000)
        self.assertEqual(trade_risk_pct(cfg, 9500, 10000, 8000), (0.125, True))

    def test_mode_changes_daily_permissions(self):
        cfg, mode = combo_config(), RiskMode()
        self.assertIsNone(daily_ok(cfg, 10000, -160, 0, 20))
        mode.select(cfg, 9700, 10000)
        self.assertEqual(daily_ok(cfg, 10000, -160, 0, 20), 'daily loss limit')

    def test_restart_preserves_conservative_mode_between_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            eng = ShadowEngine.__new__(ShadowEngine)
            eng.cfg, eng.risk_mode = combo_config(), RiskMode()
            eng.out = Path(folder)
            eng.state_file = eng.out / 'shadow_state.json'
            eng.closed_file = eng.out / 'closed_trades.csv'
            eng.log = lambda *a: None
            eng.state_file.write_text(json.dumps(dict(version=STATE_VERSION, equity=9800,
                peak=10000, kill=False, risk_mode='conservative')))
            eng._load_state()
            eng.risk_mode.select(eng.cfg, eng.equity, eng.peak)
            self.assertFalse(eng.risk_mode.aggressive)
            self.assertEqual(eng.cfg.risk_pct, 0.25)

    def test_pending_fill_uses_current_mode_and_kill_blocks_fill(self):
        eng = ShadowEngine.__new__(ShadowEngine)
        eng.cfg, eng.risk_mode = combo_config(), RiskMode()
        eng.equity, eng.peak, eng.kill = 9700, 10000, False
        eng.month, eng.month_start = '2026-10', 10000
        eng.day_start, eng.day_pnl, eng.open, eng.filled_now = {}, {}, [], set()
        eng.log = lambda *a: None
        eng._log_decision = lambda *a: None
        eng._remember_traded_zone = lambda *a: None
        order = dict(bar_time=pd.Timestamp('2026-10-06', tz='UTC'), zone='z',
                     symbol='EURUSD', usd_side=-1, bull=True, entry=1.1, risk=0.001,
                     exit='tp1', lots=0.5, risk_usd=50, risk_pct=0.5)
        self.assertTrue(eng._fill('EURUSD', order, 'candle'))
        self.assertEqual(eng.open[0]['lots'], 0.24)
        self.assertEqual(eng.open[0]['risk_pct'], 0.25)
        self.assertEqual(order['lots'], 0.5)  # Existing order data is not mutated.
        eng.equity = 8900
        self.assertFalse(eng._fill('GBPUSD', order, 'candle'))
        self.assertTrue(eng.kill)


if __name__ == '__main__':
    unittest.main()
