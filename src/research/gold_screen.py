r"""
GOLD STRATEGY SCREEN (14 Oct 2026) - 3 known intraday ideas, ONE fixed setting each, XAUUSD M5 bid + bar spread.
Costs: $7/lot round trip (1 lot = 100 oz), $0.05 slip on stop exits; buy fills at ask, sell exits at ask; SL first in a bar.
  G1 ASIA_SWEEP  Asia range 19:00-00:00 NY. In London killzone 02:00-05:00 NY: price trades beyond the Asia high (low)
                 and an M5 bar CLOSES back inside -> SELL (BUY) next open. SL = sweep extreme + 0.1 x range,
                 TP = 2R, time exit 11:00 NY. Max 1 per day.
  G2 NY_MOM      Move 03:00 -> 08:00 NY > 0.3 x D1-ATR(14) -> trade WITH it at 08:00, SL 0.4 x D1-ATR, exit 12:00 NY.
  G3 CPR_PULL    Daily CPR from yesterday (P=(H+L+C)/3, BC=(H+L)/2, TC=2P-BC). Open of day above the CPR top ->
                 first touch of the CPR top between 03:00-12:00 NY = BUY (mirror SELL); SL = other CPR side - 0.1 x D1-ATR,
                 TP 2R, exit 16:00 NY. Max 1 per day.
SCREEN PASS (fixed before running): net PF >= 1.15 in 2016-2020 AND >= 1.10 in 2021-2026 AND >= 100 trades/year.
M5 is optimistic -> a pass only earns the exact tick test.
Chalana: python src\research\gold_screen.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
NY = "America/New_York"
COMM, SLIP, OZ = 7.0, 0.05, 100.0


def sim(O, H, L, C, S, k, bull, sl, tp, k_end):
    entry = O[k] + S[k] if bull else O[k]
    for j in range(k, min(k_end, len(O) - 1) + 1):
        if bull:
            if L[j] <= sl:
                return entry, (min(sl, O[j]) if j > k else sl), "sl"
            if tp is not None and H[j] >= tp:
                return entry, tp, "tp"
        else:
            if H[j] + S[j] >= sl:
                return entry, (max(sl, O[j] + S[j]) if j > k else sl), "sl"
            if tp is not None and L[j] + S[j] <= tp:
                return entry, tp, "tp"
    j = min(k_end, len(O) - 1)
    return entry, (C[j] if bull else C[j] + S[j]), "time"


def book(rows, name, t, bull, entry, exitp, sl, why):
    risk = abs(entry - sl)
    if risk <= 0:
        return
    r = ((exitp - entry) if bull else (entry - exitp)) / risk
    r -= COMM / (risk * OZ) + (SLIP / risk if why == "sl" else 0.0)
    rows.append(dict(strategy=name, time=t, bull=bull, r=r, why=why, risk_usd_oz=risk))


def run():
    m = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["open", "high", "low", "close", "spread"])
    m["spread"] = m.spread.clip(lower=0)
    d1 = pd.read_parquet(ROOT / "data/processed/XAUUSD_D1.parquet")
    O, H, L, C, S = (m[c].to_numpy() for c in ("open", "high", "low", "close", "spread"))
    tv = m.index.values
    pc = d1.close.shift()
    atr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs())).rolling(14).mean().to_numpy()
    ct = pd.DatetimeIndex(d1.close_time).tz_convert(None).values
    dh, dl, dc = d1.high.to_numpy(), d1.low.to_numpy(), d1.close.to_numpy()
    idx = lambda ts: int(np.searchsorted(tv, np.datetime64(ts.tz_convert("UTC").tz_localize(None), "ns")))  # noqa: E731
    rows = []
    for day in pd.date_range("2016-01-18", "2026-09-24", freq="B", tz=NY):
        T = {h: day + pd.Timedelta(hours=h) for h in (2, 3, 5, 8, 11, 12, 16)}
        a19 = idx(day - pd.Timedelta(hours=5)); a00 = idx(day)
        i2, i3, i5, i8, i11, i12, i16 = (idx(T[h]) for h in (2, 3, 5, 8, 11, 12, 16))
        if i16 >= len(O) or a00 - a19 < 30 or i16 - i3 < 100:
            continue
        di = int(np.searchsorted(ct, np.datetime64(day.tz_convert("UTC").tz_localize(None), "ns"), side="right")) - 1
        if di < 15 or not np.isfinite(atr[di]):
            continue
        A = atr[di]
        # G1 Asia sweep reversal
        ahi, alo = H[a19:a00].max(), L[a19:a00].min(); rg = ahi - alo
        swept_hi = swept_lo = None
        for j in range(i2, i5):
            if H[j] > ahi:
                swept_hi = max(swept_hi or H[j], H[j])
            if L[j] < alo:
                swept_lo = min(swept_lo or L[j], L[j])
            if swept_hi is not None and C[j] < ahi and C[j] > alo:
                sl = swept_hi + 0.1 * rg; e0 = O[j + 1]; tp = e0 - 2 * (sl - e0)
                e, x, why = sim(O, H, L, C, S, j + 1, False, sl, tp, i11)
                book(rows, "G1_ASIA_SWEEP", m.index[j + 1], False, e, x, sl, why); break
            if swept_lo is not None and C[j] > alo and C[j] < ahi:
                sl = swept_lo - 0.1 * rg; e0 = O[j + 1] + S[j + 1]; tp = e0 + 2 * (e0 - sl)
                e, x, why = sim(O, H, L, C, S, j + 1, True, sl, tp, i11)
                book(rows, "G1_ASIA_SWEEP", m.index[j + 1], True, e, x, sl, why); break
        # G2 London->NY momentum
        mv = C[i8 - 1] - O[i3]
        if abs(mv) >= 0.3 * A:
            bull = mv > 0; e0 = O[i8] + S[i8] if bull else O[i8]
            sl = e0 - 0.4 * A if bull else e0 + 0.4 * A
            e, x, why = sim(O, H, L, C, S, i8, bull, sl, None, i12 - 1)
            book(rows, "G2_NY_MOM", m.index[i8], bull, e, x, sl, why)
        # G3 CPR pullback
        P = (dh[di] + dl[di] + dc[di]) / 3; BC = (dh[di] + dl[di]) / 2; TC = 2 * P - BC
        top, bot = max(TC, BC), min(TC, BC); op = O[a19]
        if op > top or op < bot:
            bull = op > top
            for j in range(i3, i12):
                if (bull and L[j] <= top) or (not bull and H[j] + S[j] >= bot):
                    e0 = top if bull else bot
                    sl = bot - 0.1 * A if bull else top + 0.1 * A
                    risk = abs(e0 - sl); tp = e0 + 2 * risk if bull else e0 - 2 * risk
                    # limit fill at the CPR edge inside bar j; then manage from bar j+1 (SL-first check in bar j too)
                    if (bull and L[j] <= sl) or (not bull and H[j] + S[j] >= sl):
                        book(rows, "G3_CPR_PULL", m.index[j], bull, e0, sl, sl, "sl"); break
                    e, x, why = sim(O, H, L, C, S, j + 1, bull, sl, tp, i16 - 1)
                    book(rows, "G3_CPR_PULL", m.index[j], bull, e0, x, sl, why); break
    d = pd.DataFrame(rows)
    d.to_parquet(ROOT / "data/research/gold_screen.parquet")
    return d


def pf(r):
    return round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan


def report(d):
    d["year"] = pd.DatetimeIndex(d.time).year
    out = []
    for st, g in d.groupby("strategy"):
        Lr, Tr = g[g.year <= 2020], g[g.year >= 2021]
        n_yr = len(g) / 10.7
        ok = pf(Lr.r) >= 1.15 and pf(Tr.r) >= 1.10 and n_yr >= 100
        out.append(dict(strategy=st, trades_yr=round(n_yr), win=round(float((g.r > 0).mean()) * 100), pf_2016_20=pf(Lr.r),
                        pf_2021_26=pf(Tr.r), total_R=round(g.r.sum()), median_risk_usd=round(g.risk_usd_oz.median(), 2),
                        PASS="PASS" if ok else "FAIL"))
    print(pd.DataFrame(out).to_string(index=False))
    print("\nPF per year:"); print(d.groupby(["strategy", "year"]).r.apply(pf).unstack(0).to_string())


if __name__ == "__main__":
    report(run())
