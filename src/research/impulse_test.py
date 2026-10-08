r"""
IMPULSE DISPLACEMENT TEST (his way of seeing a zone), all on REAL TICKS.

Old bot: displacement = zone candle low -> close of the FIRST candle that breaks a swing (often ~2x ATR).
His way: displacement = the WHOLE move away from the zone (zone low -> top of the impulse) before price comes back.

Rules (fixed before running):
  - zones = last opposite candle before a BOS (same detector, swing_n = 1 / 2 / 3), M15
  - after the BOS we follow the move. The zone becomes VALID only when the move has gone N x ATR
    (ATR of the zone candle) away from the zone's far edge. N = 2, 3, 4, 5, 6.
    Order is placed only after that candle CLOSES (no lookahead).
  - if price comes back to the entry BEFORE the zone is valid -> zone skipped (move was not strong)
  - zone life 7 days after it becomes valid; first touch only
  - entry = body edge (limit), SL = wick + 0.1 ATR (same as before), exits on ticks: 1:1, 1:2, trail
  - costs: $5 / lot + 0.2 pip on stop exits
Steps:  python src\research\impulse_test.py zones            (once: zone lists for swing 1/2/3)
        python src\research\impulse_test.py run EURUSD 2019 (one pair-year, all swings + N)
        python src\research\impulse_test.py report
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr, find_orderflow_zones  # noqa: E402

OUT = ROOT / "data/research/impulse_test"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
SWINGS = [1, 2, 3]
NS = [2, 3, 4, 5, 6]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
LIFE = pd.Timedelta(days=7)
COMM, SLIP = 5.0, 0.2


def m15(sym):
    m = pd.read_parquet(ROOT / f"data/processed/{sym}_M15.parquet")
    m["atr"] = atr(m)
    return m


def build_zones():
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in PAIRS:
        m = m15(sym)
        for s in SWINGS:
            z = find_orderflow_zones(m, swing_n=s, min_disp_atr=0.0)
            z.to_parquet(OUT / f"zones_{sym}_s{s}.parquet")
            print(sym, "swing", s, len(z), "zones", flush=True)


def run(sym, year):
    m = m15(sym)
    hi, lo, cl = m["high"].to_numpy(), m["low"].to_numpy(), m["close"].to_numpy()
    idx = m.index
    ctime = (m["close_time"] if "close_time" in m else m.index.to_series() + pd.Timedelta(minutes=15))
    cms = (pd.DatetimeIndex(ctime).as_unit("ns").asi8 // 1_000_000)
    rows = []
    exit_cache = {}
    for s in SWINGS:
        z = pd.read_parquet(OUT / f"zones_{sym}_s{s}.parquet")
        z = z[pd.to_datetime(z["bos_time"]).dt.year == year].sort_values("bos_time")
        for r in z.itertuples(index=False):
            bull = r.direction == "bull"
            a = float(r.atr)
            if not a > 0:
                continue
            entry = r.body_top if bull else r.body_bottom
            sl = r.bottom - 0.1 * a if bull else r.top + 0.1 * a
            i = int(idx.searchsorted(pd.Timestamp(r.bos_time)))
            j = min(i + 7 * 96, len(m))
            far = r.bottom if bull else r.top                  # zone's far edge (wick)
            move = ((hi[i:j] - far) if bull else (far - lo[i:j])) / a
            best = np.maximum.accumulate(move)
            touch = (lo[i:j] <= entry) if bull else (hi[i:j] >= entry)
            touch[0] = False                                   # BOS candle itself moves away
            t_idx = int(np.argmax(touch)) if touch.any() else len(touch)
            for n in NS:
                ok = best >= n
                k = int(np.argmax(ok)) if ok.any() else len(ok)
                base = dict(symbol=sym, swing=s, n=n, direction=r.direction, zone_time=r.zone_candle_time,
                            entry=entry, sl=sl, atr=a, sl_pips=abs(entry - sl) / PIP[sym])
                if k >= len(ok) or k >= t_idx:                 # never strong enough before price came back
                    rows.append(dict(base, status="not_valid")); continue
                act_ms = int(cms[i + k])
                f = te.find_fill(sym, bull, entry, act_ms, act_ms + int(LIFE.total_seconds() * 1000))
                if f is None or f == "missing":
                    rows.append(dict(base, status="no_fill" if f is None else "missing")); continue
                out = dict(base, status="ok", fill_ms=f, valid_ms=act_ms)
                key = (bull, entry, sl, f)
                if key not in exit_cache:
                    exit_cache[key] = {p: te.run_exit(sym, bull, entry, sl, f, p) for p in ("tp1", "tp2", "trail")}
                for p, x in exit_cache[key].items():
                    out[f"r_{p}"], out[f"why_{p}"] = x["r"], x["reason"]
                rows.append(out)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / f"res_{sym}_{year}.parquet")
    print(sym, year, d.status.value_counts().to_dict(), flush=True)


def report():
    d = pd.concat([pd.read_parquet(f) for f in OUT.glob("res_*.parquet")], ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["year"] = pd.to_datetime(d.fill_ms, unit="ms").dt.year
    per_lot = np.where(d.symbol.str.endswith("USD"), (d.entry - d.sl).abs() * 1e5, (d.entry - d.sl).abs() * 1e5 / d.entry)
    cost = COMM / per_lot
    slip = SLIP * d.symbol.map(PIP) / (d.entry - d.sl).abs()
    for p in ("tp1", "tp2", "trail"):
        d[f"n_{p}"] = d[f"r_{p}"] - cost - np.where(d[f"why_{p}"].isin(["sl", "trail", "be"]), slip, 0)
    pf = lambda r: round(r[r > 0].sum() / -r[r < 0].sum(), 2) if (r < 0).any() else np.nan  # noqa: E731
    days = (pd.to_datetime(d.fill_ms.max(), unit="ms") - pd.to_datetime(d.fill_ms.min(), unit="ms")).days * 5 / 7
    rows = []
    for (s, n), g in d.groupby(["swing", "n"]):
        a, b = g[g.year <= 2020], g[g.year >= 2021]
        rows.append(dict(swing=s, N_atr=n, trades=len(g), per_day=round(len(g) / days / 3, 2),
                         win_1to1=round(100 * (g.r_tp1 > 0).mean(), 1), sl_pips=round(g.sl_pips.median(), 1),
                         pf_1to1=pf(g.n_tp1), pf_1to2=pf(g.n_tp2), pf_trail=pf(g.n_trail),
                         gross_pf_1to1=pf(g.r_tp1), R_1to1=round(g.n_tp1.sum()),
                         pf_1to1_2016_20=pf(a.n_tp1), pf_1to1_2021_26=pf(b.n_tp1),
                         pf_trail_2016_20=pf(a.n_trail), pf_trail_2021_26=pf(b.n_trail)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "report.csv", index=False)
    print(t.to_string(index=False))
    return d, t


if __name__ == "__main__":
    {"zones": lambda: build_zones(), "run": lambda: run(sys.argv[2], int(sys.argv[3])),
     "report": lambda: report()}[sys.argv[1]]()
