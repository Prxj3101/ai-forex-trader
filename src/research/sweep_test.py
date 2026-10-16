"""
Sweep + reclaim test (16 Oct 2026). Swing zone test ne dikhaya: swing high / low ka wick zyada tar TOOT-TA hai
(1:1 win < 50%). Toh ulta: level sweep ho, price wapas andar CLOSE kare, tab entry.

RULES (run se PEHLE fix):
  Levels:
    - SWING: fractal swing high / low, M15 aur H1, n = 3 / 5. Active jab swing confirm ho (right side n candles band)
      AUR price level se K ATR door ja chuka ho (K = 1 / 2, us TF ka ATR14). Life 40 trading din.
    - ASIA: Asia range high / low (20:00-24:00 NY), active 00:00-16:00 NY usi din. K = 0.5 / 1 H1 ATR.
    - PDH / PDL: pichle trading din (17:00 NY roll) ka high / low, active poora agla trading din. K = 0.5 / 1 H1 ATR.
  Sweep: pehli M5 candle jiska high level ke UPAR (high level) / low NEECHE (low level). Har level sirf ek baar.
  Reclaim: usi candle ya agli 2 M5 candles mein se kisi ka CLOSE wapas level ke andar. Sweep extreme
    (poke se reclaim tak ka max high / min low) level se 1.5 ATR se zyada door -> discard (asli breakout).
  Entry: reclaim candle ke close pe market (SELL = bid close, BUY = close + spread). SL = extreme +- 0.1 ATR.
    SL 0.25-3 ATR. Exits 1:1 / 1:2 / trail (management.run_trade_full, agli candle se), max 3 din.
  Costs: $5/lot + 0.2 pip (walkforward.add_costs). EURUSD / USDJPY / GBPUSD, 2016-20 aur 2021-26 alag.
  Verdict: kaam ka tabhi jab koi variant NET PF > 1.05 DONO periods mein (1:1 ya 1:2), pados ke variants > 1,
    aur 3 mein se kam se kam 2 pairs > 1.

Chalana: python src/research/sweep_test.py run --symbol EURUSD ; python src/research/sweep_test.py report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.smc import atr, find_swings  # noqa: E402
from src.research.management import POLICIES, run_trade_full  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402
from src.research.zone_dataset_v2 import asian_ranges  # noqa: E402

OUT = ROOT / "data" / "research" / "sweep_test"
NY = "America/New_York"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
LIFE, HOLD, BPD = 40, 3, 288
RECLAIM_BARS, MAX_EXT, BUF = 3, 1.5, 0.1
POL = {p.name: p for p in POLICIES}
LABELS = {"r_tp1": POL["tp1"], "r_tp2": POL["tp2"], "r_trail": POL["trail1"]}


class M5:
    def __init__(self, df):
        self.idx = df.index
        self.lo, self.hi, self.cl, self.sp = (df[c].to_numpy() for c in ["low", "high", "close", "spread"])


def one(m, level, is_high, s, e, a, K):
    """Ek level -> trade dict ya None."""
    lo, hi, cl, sp = m.lo, m.hi, m.cl, m.sp
    if s >= e or not np.isfinite(a) or a <= 0:
        return None
    poke_arr = (hi[s:e] > level) if is_high else (lo[s:e] < level)
    pk = np.nonzero(poke_arr)[0]
    if pk.size == 0:
        return None
    p = s + int(pk[0])
    if K > 0:
        away = (lo[s:p] <= level - K * a) if is_high else (hi[s:p] >= level + K * a)
        if not away.any():
            return None
    for k in range(p, min(p + RECLAIM_BARS, e)):
        back = cl[k] < level if is_high else cl[k] > level
        if back:
            break
    else:
        return None
    ext = hi[p:k + 1].max() if is_high else lo[p:k + 1].min()
    if abs(ext - level) > MAX_EXT * a:
        return None
    bull = not is_high
    entry = cl[k] + sp[k] if bull else cl[k]
    stop = ext - BUF * a if bull else ext + BUF * a
    risk = (entry - stop) if bull else (stop - entry)
    if not (0.25 * a <= risk <= 3 * a):
        return None
    g = min(k + 1 + HOLD * BPD, len(hi))
    row = dict(direction="bull" if bull else "bear", level=level, fill_time=m.idx[k], entry=entry, sl=stop,
               risk=risk, risk_atr=risk / a, age_days=(p - s) / BPD, sweep_atr=abs(ext - level) / a)
    for col, pol in LABELS.items():
        r, why, _ = run_trade_full(pol, bull, entry, risk, hi[k + 1:g], lo[k + 1:g], cl[k + 1:g], sp[k + 1:g],
                                   fill_bar_partial=False)
        row[col], row[col + "_exit"] = float(r), why
    return row


def run(sym):
    m5df = load(sym, "M5")
    m = M5(m5df)
    h1 = load(sym, "H1")
    h1_atr = atr(h1)
    rows = []

    def add(r, **meta):
        if r:
            r.update(symbol=sym, **meta)
            rows.append(r)

    # SWING levels
    for tf in ("M15", "H1"):
        df = load(sym, tf)
        a = atr(df).to_numpy()
        h, l = df["high"].to_numpy(), df["low"].to_numpy()
        for n in (3, 5):
            sh, sl_ = find_swings(df, n)
            for j in np.nonzero(sh | sl_)[0]:
                if j + n + 1 >= len(df):
                    continue
                s = int(m.idx.searchsorted(df.index[j + n + 1]))
                e = min(s + LIFE * BPD, len(m.hi))
                for K in (1.0, 2.0):
                    if sh[j]:
                        add(one(m, h[j], True, s, e, a[j], K), kind=f"swing_{tf}", n=n, K=K)
                    if sl_[j]:
                        add(one(m, l[j], False, s, e, a[j], K), kind=f"swing_{tf}", n=n, K=K)
        print(f"{sym} swing {tf}: {len(rows):,}")

    # ASIA + PDH/PDL (H1 ATR = last closed H1 before window start)
    ny = m5df.index.tz_convert(NY)
    def h1a(ts):
        i = h1.index.searchsorted(ts) - 1
        return float(h1_atr.iloc[i]) if i >= 0 else np.nan

    asia = asian_ranges(m5df)
    for d, r in asia.iterrows():
        t0 = pd.Timestamp(d).tz_localize(NY)
        s, e = int(m.idx.searchsorted(t0)), int(m.idx.searchsorted(t0 + pd.Timedelta(hours=16)))
        a = h1a(t0)
        for K in (0.5, 1.0):
            add(one(m, r.asia_high, True, s, e, a, K), kind="asia", n=0, K=K)
            add(one(m, r.asia_low, False, s, e, a, K), kind="asia", n=0, K=K)
    tday = (ny + pd.Timedelta(hours=7)).date          # 17:00 NY roll
    g = m5df.groupby(tday)
    daily = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min(),
                          "start": pd.Series(m.idx, index=m.idx).groupby(tday).min()})
    for i in range(1, len(daily)):
        prev, cur = daily.iloc[i - 1], daily.iloc[i]
        s = int(m.idx.searchsorted(cur.start))
        e = int(m.idx.searchsorted(daily.iloc[i + 1].start)) if i + 1 < len(daily) else len(m.hi)
        a = h1a(cur.start)
        for K in (0.5, 1.0):
            add(one(m, prev.hi, True, s, e, a, K), kind="pd", n=0, K=K)
            add(one(m, prev.lo, False, s, e, a, K), kind="pd", n=0, K=K)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{sym}.parquet"
    d.to_parquet(f)
    print(f"saved {f.relative_to(ROOT)} ({len(d):,} trades)")


def pf(r):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def report():
    from src.ml.walkforward import add_costs, COMMISSION, SLIP_PIPS
    d = pd.concat([pd.read_parquet(OUT / f"{s}.parquet") for s in SYMS], ignore_index=True)
    d = add_costs(d, COMMISSION, SLIP_PIPS)
    d["period"] = np.where(d["fill_time"].dt.year <= 2020, "2016-20", "2021-26")
    keys = ["kind", "n", "K"]
    rows = []
    for k, g in d.groupby(keys):
        row = dict(zip(keys, k), trades=len(g), per_year=round(len(g) / 10.75),
                   win_1to1=round(100 * float((g["r_tp1"] > 0).mean()), 1), gross_1to1=pf(g["r_tp1_gross"]))
        for per, gp in g.groupby("period"):
            for c, lab in (("r_tp1", "1to1"), ("r_tp2", "1to2"), ("r_trail", "trail")):
                row[f"{lab}_{per}"] = pf(gp[c])
        for s, gs in g.groupby("symbol"):
            row[f"1to1_{s}"] = pf(gs["r_tp1"])
        rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "report.csv", index=False)
    pd.set_option("display.width", 250)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    r = sp_.add_parser("run")
    r.add_argument("--symbol", nargs="+", default=SYMS)
    sp_.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        for s in a.symbol:
            run(s)
    else:
        report()
