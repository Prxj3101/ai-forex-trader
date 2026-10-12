r"""
HIGH-FREQUENCY STRATEGY SCREEN (11 Oct 2026) - cheap M5 screen of 3 NEW strategy families (no tuning: ONE fixed setting each).
Pairs EURUSD / GBPUSD / USDJPY, M5 bid bars + bar spread (buy fills at ask = bid + spread, sell exits at ask),
SL first if SL and TP in the same bar, $5/lot commission + 0.2 pip slip on stop exits. One open trade per pair per strategy.
  S1 ASIA_MR   Asian session mean reversion (19:00-02:00 NY): M5 close outside Bollinger(20, 2.5) -> fade at next open;
               TP = middle band at signal, SL = 1.5 x that distance, skip if distance < 3 pips, time exit 02:00 NY.
  S2 LDN_MOM   London->NY intraday momentum: move 03:00->08:00 NY > +0.3 x D1-ATR(14) -> BUY at 08:00 (< -0.3 -> SELL);
               SL 0.4 x D1-ATR, no TP, exit 12:00 NY.
  S3 H1_PULL   H1 trend pullback: EMA50 > EMA200, H1 low <= EMA20, close > EMA20 and green -> BUY next H1 open (mirror SELL);
               SL = lowest low of last 3 H1 bars - 0.2 x H1-ATR(14), TP 2R, max hold 48 h.
SCREEN PASS (fixed before running): net PF >= 1.15 in 2016-2020 AND >= 1.10 in 2021-2026 AND >= 1 trade/day over the 3 pairs.
M5 screen is optimistic (~+0.03-0.3 PF vs ticks, worst for fast trades) -> a pass only earns a tick test, nothing more.
Chalana: python src\research\hf_screen.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
NY = "America/New_York"


def bars(sym, tf):
    return pd.read_parquet(ROOT / f"data/processed/{sym}_{tf}.parquet", columns=["open", "high", "low", "close", "spread", "close_time"])


def sim(O, H, L, C, S, k, bull, sl, tp, k_end):
    """Enter at open of bar k (buy at ask). Returns (entry, exit_px, why, k_exit). Exit side: buy->bid, sell->ask."""
    entry = O[k] + S[k] if bull else O[k]
    for j in range(k, min(k_end, len(O) - 1) + 1):
        if bull:
            if L[j] <= sl:
                return entry, min(sl, O[j]) if j > k else sl, "sl", j
            if tp is not None and H[j] >= tp:
                return entry, tp, "tp", j
        else:
            if H[j] + S[j] >= sl:
                return entry, max(sl, O[j] + S[j]) if j > k else sl, "sl", j
            if tp is not None and L[j] + S[j] <= tp:
                return entry, tp, "tp", j
    j = min(k_end, len(O) - 1)
    return entry, (C[j] if bull else C[j] + S[j]), "time", j


def book(sym, rows, name, t, bull, entry, exitp, sl, why):
    risk = abs(entry - sl)
    if risk <= 0:
        return
    r = ((exitp - entry) if bull else (entry - exitp)) / risk
    per_lot = risk * 1e5 if sym.endswith("USD") else risk * 1e5 / entry
    r -= 5.0 / per_lot + (0.2 * PIP[sym] / risk if why == "sl" else 0.0)
    rows.append(dict(strategy=name, symbol=sym, time=t, bull=bull, r=r, why=why, risk_pips=risk / PIP[sym]))


def s1(sym, m, rows):
    O, H, L, C, S = (m[c].to_numpy() for c in ("open", "high", "low", "close", "spread"))
    mid = m.close.rolling(20).mean().to_numpy(); sd = m.close.rolling(20).std().to_numpy()
    ny = m.index.tz_convert(NY); hr = ny.hour.to_numpy()
    asia = (hr >= 19) | (hr < 2)
    # time exit index: first bar at/after 02:00 NY following the signal
    two = np.flatnonzero((hr == 2) & (ny.minute.to_numpy() == 0))
    i, n = 20, len(m) - 2
    while i < n:
        if asia[i] and np.isfinite(sd[i]) and (C[i] < mid[i] - 2.5 * sd[i] or C[i] > mid[i] + 2.5 * sd[i]):
            bull = C[i] < mid[i]; k = i + 1
            entry0 = O[k] + S[k] if bull else O[k]
            d = (mid[i] - entry0) if bull else (entry0 - mid[i])
            if d >= 3 * PIP[sym] and asia[k]:
                tp = mid[i]; sl = entry0 - 1.5 * d if bull else entry0 + 1.5 * d
                ke = two[np.searchsorted(two, k)] if np.searchsorted(two, k) < len(two) else len(m) - 1
                ke = min(ke, k + 12 * 8)
                e, x, why, j = sim(O, H, L, C, S, k, bull, sl, tp, ke)
                book(sym, rows, "S1_ASIA_MR", m.index[k], bull, e, x, sl, why)
                i = j + 1; continue
        i += 1


def s2(sym, m, d1, rows):
    O, H, L, C, S = (m[c].to_numpy() for c in ("open", "high", "low", "close", "spread"))
    tv = m.index.values
    pc = d1.close.shift()
    atr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs())).rolling(14).mean().to_numpy()
    ct = pd.DatetimeIndex(d1.close_time).tz_convert(None).values
    for day in pd.date_range("2016-01-11", "2026-09-24", freq="B", tz=NY):
        t3, t8, t12 = (day + pd.Timedelta(hours=h) for h in (3, 8, 12))
        a, b, c = np.searchsorted(tv, [np.datetime64(x.tz_convert("UTC").tz_localize(None), "ns") for x in (t3, t8, t12)])
        if c >= len(O) or b - a < 50 or c - b < 40:
            continue
        di = np.searchsorted(ct, np.datetime64(t3.tz_convert("UTC").tz_localize(None), "ns"), side="right") - 1
        if di < 15 or not np.isfinite(atr[di]):
            continue
        mv = C[b - 1] - O[a]
        if abs(mv) < 0.3 * atr[di]:
            continue
        bull = mv > 0
        entry0 = O[b] + S[b] if bull else O[b]
        sl = entry0 - 0.4 * atr[di] if bull else entry0 + 0.4 * atr[di]
        e, x, why, j = sim(O, H, L, C, S, b, bull, sl, None, c - 1)
        book(sym, rows, "S2_LDN_MOM", m.index[b], bull, e, x, sl, why)


def s3(sym, m, h1, rows):
    O, H, L, C, S = (m[c].to_numpy() for c in ("open", "high", "low", "close", "spread"))
    tv = m.index.values
    e20, e50, e200 = (h1.close.ewm(span=s, adjust=False).mean().to_numpy() for s in (20, 50, 200))
    pc = h1.close.shift()
    atr = np.maximum(h1.high - h1.low, np.maximum((h1.high - pc).abs(), (h1.low - pc).abs())).rolling(14).mean().to_numpy()
    ho, hh, hl, hc = (h1[c].to_numpy() for c in ("open", "high", "low", "close"))
    hct = pd.DatetimeIndex(h1.close_time).tz_convert(None).values
    busy_until = np.datetime64("1970-01-01")
    for i in range(210, len(h1) - 1):
        if hct[i] < busy_until:
            continue
        up = e50[i] > e200[i] and hl[i] <= e20[i] and hc[i] > e20[i] and hc[i] > ho[i]
        dn = e50[i] < e200[i] and hh[i] >= e20[i] and hc[i] < e20[i] and hc[i] < ho[i]
        if not (up or dn) or not np.isfinite(atr[i]):
            continue
        k = np.searchsorted(tv, hct[i])                       # first M5 bar starting at the H1 close
        if k >= len(O) - 2:
            break
        bull = up
        entry0 = O[k] + S[k] if bull else O[k]
        sl = hl[i - 2:i + 1].min() - 0.2 * atr[i] if bull else hh[i - 2:i + 1].max() + 0.2 * atr[i]
        risk = abs(entry0 - sl)
        if (bull and sl >= entry0) or (not bull and sl <= entry0) or risk < 3 * PIP[sym]:
            continue
        tp = entry0 + 2 * risk if bull else entry0 - 2 * risk
        e, x, why, j = sim(O, H, L, C, S, k, bull, sl, tp, k + 12 * 48)
        book(sym, rows, "S3_H1_PULL", m.index[k], bull, e, x, sl, why)
        busy_until = tv[j]


def run():
    rows = []
    for s in PAIRS:
        m = bars(s, "M5"); m["spread"] = m.spread.clip(lower=0)
        s1(s, m, rows); s2(s, m, bars(s, "D1"), rows); s3(s, m, bars(s, "H1"), rows)
        print(s, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(ROOT / "data/research/hf_screen.parquet")
    return d


def pf(r):
    return round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan


def report(d):
    d["year"] = pd.DatetimeIndex(d.time).year
    days = (pd.Timestamp("2026-09-24") - pd.Timestamp("2016-01-04")).days * 5 / 7
    out = []
    for st, g in d.groupby("strategy"):
        L, T = g[g.year <= 2020], g[g.year >= 2021]
        ok = pf(L.r) >= 1.15 and pf(T.r) >= 1.10 and len(g) / days >= 1
        out.append(dict(strategy=st, trades_per_day=round(len(g) / days, 2), win=round(float((g.r > 0).mean()) * 100),
                        avg_R=round(g.r.mean(), 3), pf_2016_20=pf(L.r), pf_2021_26=pf(T.r), total_R=round(g.r.sum()),
                        median_sl_pips=round(g.risk_pips.median(), 1), PASS="PASS" if ok else "FAIL"))
    print(pd.DataFrame(out).to_string(index=False))
    print("\nPF per pair:"); print(d.groupby(["strategy", "symbol"]).r.apply(pf).unstack().to_string())
    print("\nPF per year:"); print(d.groupby(["strategy", "year"]).r.apply(pf).unstack(0).to_string())


if __name__ == "__main__":
    report(run())
