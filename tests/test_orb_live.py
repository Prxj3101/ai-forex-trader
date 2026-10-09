"""ORB live engine (src/live/orb.py): time windows (DST) + replay must match the backtest exactly."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live import orb  # noqa: E402

NY = "America/New_York"


def _ny(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(NY)


@pytest.mark.parametrize("day", ["2026-03-09", "2026-03-06", "2026-11-02", "2026-07-15"])
def test_windows_are_ny_wall_clock(day):
    rs, re_, we, eod = orb.windows(day, "ASIA")
    assert (_ny(rs).hour, _ny(re_).hour, _ny(we).hour, _ny(eod).hour) == (19, 2, 11, 16)
    assert str(_ny(re_).date()) == day
    rs, re_, _, _ = orb.windows(day, "LON30")
    assert (_ny(rs).hour, _ny(rs).minute, _ny(re_).hour, _ny(re_).minute) == (3, 0, 3, 30)


def test_replay_matches_backtest():
    need = [ROOT / "models/orb_brain.txt", ROOT / "data/research/orb_ml_features.parquet",
            ROOT / "data/ticks/store/EURUSD/2026-01.npz", ROOT / "data/processed/EURUSD_M5.parquet"]
    if not all(p.exists() for p in need):
        pytest.skip("data / model not available")
    from src.live import orb_replay_check as rc
    eng = rc.run("2026-01-12", "2026-01-24")
    t = rc.compare(eng, "2026-01-12", "2026-01-24")
    assert len(t) > 0 and float((t.live_score - t.bt_score).abs().max()) < 1e-3
    tr = pd.read_csv(rc.OUT / "orb_trades.csv")
    assert len(tr) > 0
