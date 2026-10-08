r"""
CONFIRMATION ENTRY TEST (real ticks). Instead of a blind limit at the zone:
  1. zone valid after the impulse reached N x ATR (same as impulse_test)
  2. price TOUCHES the zone entry (M5 candle)
  3. CONFIRMATION = an M5 candle CLOSES back outside the zone in trade direction
     (BUY: close above zone top; SELL: close below zone bottom) within 3 hours of the touch,
     before any M5 close beyond the far wick (= zone broken -> skip)
  4. MARKET entry at the first tick after that candle closes (BUY ask, SELL bid)
  5. SL A = zone wick + 0.1 ATR ; SL B = extreme since touch + 0.1 ATR ; exits on ticks (1:1, 1:2, trail)
Pass mark (fixed before): net PF > 1.10 and PF > 1 in both 2016-20 and 2021-26.
Chalana: python src\research\confirm_test.py EURUSD 1 2
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr  # noqa: E402

OUT = ROOT / "data/research/confirm_test"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
COMM, SLIP = 5.0, 0.2
CONFIRM_BARS = 36            # 3 hours of M5


def run(sym, swing, n, year):
    m = pd.read_parquet(ROOT / f"data/processed/{sym}_M15.parquet"); m["atr"] = atr(m)
    hi, lo = m["high"].to_numpy(), m["low"].to_numpy()
    idx15 = m.index
    m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["high", "low", "close"])
    h5, l5, c5 = m5["high"].to_numpy(), m5["low"].to_numpy(), m5["close"].to_numpy()
    idx5 = m5.index
    z = pd.read_parquet(ROOT / f"data/research/impulse_test/zones_{sym}_s{swing}.parquet").sort_values("bos_time")
    z = z[pd.to_datetime(z.bos_time).dt.year == year]
    rows = []
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"; a = float(r.atr)
        if not a > 0:
            continue
        entry_lvl = r.body_top if bull else r.body_bottom
        far = r.bottom if bull else r.top
        near = r.top if bull else r.bottom
        i = int(idx15.searchsorted(r.bos_time)); j = min(i + 7 * 96, len(m))
        move = ((hi[i:j] - far) if bull else (far - lo[i:j])) / a
        best = np.maximum.accumulate(move)
        touch15 = (lo[i:j] <= entry_lvl) if bull else (hi[i:j] >= entry_lvl); touch15[0] = False
        t_idx = int(np.argmax(touch15)) if touch15.any() else len(touch15)
        ok = best >= n; k = int(np.argmax(ok)) if ok.any() else len(ok)
        if k >= len(ok) or k >= t_idx:
            continue
        valid_t = idx15[i + k] + pd.Timedelta(minutes=15)
        s5 = int(idx5.searchsorted(valid_t)); e5 = min(s5 + 7 * 288, len(m5))
        tch = (l5[s5:e5] <= entry_lvl) if bull else (h5[s5:e5] >= entry_lvl)
        if not tch.any():
            continue
        t5 = s5 + int(np.argmax(tch))
        w = slice(t5, min(t5 + CONFIRM_BARS, len(m5)))
        cc = c5[w]
        conf = (cc > near) if bull else (cc < near)
        broke = (cc < far) if bull else (cc > far)
        ci = int(np.argmax(conf)) if conf.any() else 10**9
        bi = int(np.argmax(broke)) if broke.any() else 10**9
        base = dict(symbol=sym, swing=swing, n=n, direction=r.direction, zone_time=r.zone_candle_time, atr=a)
        if ci == 10**9 or bi <= ci:
            rows.append(dict(base, status="no_confirm")); continue
        cbar = t5 + ci
        ext = l5[t5:cbar + 1].min() if bull else h5[t5:cbar + 1].max()
        conf_ms = int((idx5[cbar] + pd.Timedelta(minutes=5)).value // 1_000_000)
        tk = te.ticks(sym, conf_ms, conf_ms + 600_000)
        if tk is None or len(tk[0]) == 0:
            rows.append(dict(base, status="missing")); continue
        e_ms = int(tk[0][0]); entry = float(tk[1][0] if bull else tk[2][0])
        out = dict(base, status="ok", entry_ms=e_ms, entry=entry)
        for tag, sl in (("A", far - 0.1 * a if bull else far + 0.1 * a), ("B", ext - 0.1 * a if bull else ext + 0.1 * a)):
            risk = (entry - sl) if bull else (sl - entry)
            if risk <= 0:
                out[f"{tag}_bad"] = True; continue
            out[f"{tag}_sl_pips"] = risk / PIP[sym]
            for p in ("tp1", "tp2", "trail"):
                x = te.run_exit(sym, bull, entry, sl, e_ms, p)
                out[f"{tag}_{p}"], out[f"{tag}_{p}_why"] = x["r"], x["reason"]
        rows.append(out)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"res_{sym}_s{swing}_n{n:g}_{year}.parquet")
    print(sym, year, d.status.value_counts().to_dict(), flush=True)


def report(d, sym):
    pf = lambda r: round(r[r > 0].sum() / -r[r < 0].sum(), 2) if (r < 0).any() else np.nan  # noqa: E731
    print(d.status.value_counts().to_dict())
    d = d[d.status == "ok"].copy(); d["year"] = pd.to_datetime(d.entry_ms, unit="ms").dt.year
    for tag in ("A", "B"):
        x = d[d[f"{tag}_sl_pips"].notna()].copy()
        risk = x[f"{tag}_sl_pips"] * PIP[sym]
        per_lot = risk * 1e5 if sym.endswith("USD") else risk * 1e5 / x.entry
        cost = COMM / per_lot; slip = SLIP * PIP[sym] / risk
        for p in ("tp1", "tp2", "trail"):
            net = x[f"{tag}_{p}"] - cost - np.where(x[f"{tag}_{p}_why"].isin(["sl", "trail", "be"]), slip, 0)
            a, b = net[x.year <= 2020], net[x.year >= 2021]
            print(f"SL {tag} {p:5s} n={len(x)} SL med {x[f'{tag}_sl_pips'].median():.1f}p win {100*(x[f'{tag}_{p}']>0).mean():.1f}% "
                  f"gross PF {pf(x[f'{tag}_{p}'])} NET PF {pf(net)} | 2016-20 {pf(a)} 2021-26 {pf(b)} | R {net.sum():.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        sym, sw, n = sys.argv[2], sys.argv[3], sys.argv[4]
        report(pd.concat([pd.read_parquet(f) for f in OUT.glob(f"res_{sym}_s{sw}_n{n}_*.parquet")]), sym)
    else:
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
