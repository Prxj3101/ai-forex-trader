r"""
ORB "IVB" TEST (video idea, 9 Oct 2026) on REAL TICKS. Rules fixed BEFORE running.

RANGES (New York time): ASIA 19:00-02:00 (window to 11:00), LON30 03:00-03:30 / LON60 03:00-04:00 (to 11:00),
  NY15 08:00-08:15 / NY30 08:00-08:30 / NY60 08:00-09:00 (to 12:00). Range hi/lo = bid ticks.
VOLUME PROFILE of the range: tick count per price bin (30 bins, mid price) -> POC, VAH, VAL (70%).
BREAK: first M5 close (bid, built from ticks) outside the range inside the window.
ENTRY after the break bar closes (BUY side; SELL is the mirror):
  BRK = market at next tick (old style, for comparison)
  VA  = limit at VAH (the value area edge)       MID = limit halfway VAH..POC       POC = limit at POC
  Limit orders live until window end; cancelled if an M5 close goes below VAL before the fill (setup invalid).
SL: VALH  = hard stop at VAL - 0.1 x range
    VALC  = video rule: exit on M5 close below VAL (backstop hard stop at range low - 0.1 x range);
            nominal risk (for lots/costs/R) = entry - (VAL - 0.1 x range), real loss can be bigger
    RNG   = hard stop at the other side of the range
TP: 1R, 2R, W05/W1/W15/W2 = range edge + 0.5/1/1.5/2 x range width ("protection level"), EOD = close 16:00 NY.
  Statistical TP: per pair+range, the W level nearest to what 65% of 2016-20 breaks reached (mfe_w quantile 0.35).
Costs $5/lot + 0.2 pip on stop exits.
SELECTION: per range, best (entry, SL, TP) combo chosen on 2016-20 ONLY, then judged on 2021-26.
PASS: 2021-26 net PF >= 1.10, >= 5 of 6 years positive, >= 2 of 3 pairs > 1, >= 150 trades, both halves
  (2021-23, 2024-26) PF > 1.
Chalana: python src\research\orb_ivb_test.py run EURUSD 2016 2017 | python src\research\orb_ivb_test.py report
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr, structure_breaks  # noqa: E402

OUT = ROOT / "data/research/orb_ivb"
NY = "America/New_York"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
RANGES = {"ASIA": ((19, 0), (2, 0), (11, 0)), "LON30": ((3, 0), (3, 30), (11, 0)), "LON60": ((3, 0), (4, 0), (11, 0)),
          "NY15": ((8, 0), (8, 15), (12, 0)), "NY30": ((8, 0), (8, 30), (12, 0)), "NY60": ((8, 0), (9, 0), (12, 0))}
EOD = (16, 0)
BAR = 300_000
ENTRIES = ("BRK", "VA", "MID", "POC")
SLS = ("VALH", "VALC", "RNG")
TPS = {"1R": ("R", 1.0), "2R": ("R", 2.0), "W05": ("W", 0.5), "W1": ("W", 1.0), "W15": ("W", 1.5), "W2": ("W", 2.0),
       "EOD": (None, None)}
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan  # noqa: E731


def ms(ts):
    return int(ts.value // 1_000_000)


def profile(mid, lo, hi, nb=30):
    cnt, edges = np.histogram(mid, bins=nb, range=(lo, hi))
    p = int(cnt.argmax()); i = j = p; tot = cnt.sum(); s = cnt[p]
    while s < 0.7 * tot and (i > 0 or j < nb - 1):
        up = cnt[j + 1] if j < nb - 1 else -1; dn = cnt[i - 1] if i > 0 else -1
        if up >= dn:
            j += 1; s += cnt[j]
        else:
            i -= 1; s += cnt[i]
    return (edges[p] + edges[p + 1]) / 2, edges[j + 1], edges[i]  # poc, vah, val


def bar_closes(t, px, t0):
    """M5 closes from ticks: (bar_end_ms, last tick index) for bars starting at t0."""
    k = (t - t0) // BAR
    last = np.nonzero(np.diff(k, append=k[-1] + 1))[0]
    return t0 + (k[last] + 1) * BAR, last


def run_day(sym, t, a, b, day, name, spec):
    (h0, m0), (h1, m1), (h2, m2) = spec
    r_start = (day - pd.Timedelta(days=1) if h0 > h1 else day) + pd.Timedelta(hours=h0, minutes=m0)
    r_end = ms((day + pd.Timedelta(hours=h1, minutes=m1)).tz_convert("UTC"))
    w_end = ms((day + pd.Timedelta(hours=h2, minutes=m2)).tz_convert("UTC"))
    eod = ms((day + pd.Timedelta(hours=EOD[0], minutes=EOD[1])).tz_convert("UTC"))
    i0, i1, i2 = np.searchsorted(t, [ms(r_start.tz_convert("UTC")), r_end, eod])
    if i1 - i0 < 50 or i2 - i1 < 50:
        return None
    rb = b[i0:i1]; hi, lo = float(rb.max()), float(rb.min()); w = hi - lo
    if w <= 0:
        return None
    poc, vah, val = profile((a[i0:i1] + rb) / 2, lo, hi)
    T, A, B = t[i1:i2], a[i1:i2], b[i1:i2]
    ends, last = bar_closes(T, B, r_end)
    ok = ends <= w_end
    cb = B[last]
    up = np.nonzero(ok & (cb > hi))[0]; dn = np.nonzero(ok & (cb < lo))[0]
    out = dict(symbol=sym, day=day.date(), range=name, hi=hi, lo=lo, w=w, poc=poc, vah=vah, val=val, broke=0)
    if not len(up) and not len(dn):
        return out
    bull = (up[0] if len(up) else 10**9) < (dn[0] if len(dn) else 10**9)
    kb = up[0] if bull else dn[0]
    brk_end = int(ends[kb])
    sgn = 1 if bull else -1
    edge = hi if bull else lo
    inv = val if bull else vah            # value area far edge (invalidation)
    near, far_poc = (vah if bull else val), poc
    xs = B if bull else A                 # exit-side price
    fill_px = A if bull else B            # entry-side price
    s_x = sgn * xs                        # signed exit price, rising = good
    # invalid closes (signed): close beyond inv
    inv_bar = (sgn * cb < sgn * inv) & (ends > brk_end)
    out.update(broke=1, direction="bull" if bull else "bear", break_ms=brk_end,
               mfe_w=float((s_x[T >= brk_end].max() - sgn * edge) / w) if (T >= brk_end).any() else np.nan)
    j0 = np.searchsorted(T, brk_end)
    if j0 >= len(T):
        return out
    levels = {"VA": near, "MID": (near + poc) / 2, "POC": poc}
    for en in ENTRIES:
        if en == "BRK":
            f = j0; entry = float(fill_px[f])
        else:
            L = levels[en]
            hit = np.nonzero((T >= brk_end) & (T < w_end) & ((fill_px <= L) if bull else (fill_px >= L)))[0]
            if not len(hit):
                out[f"{en}_fill"] = 0; continue
            f = hit[0]
            # cancelled if an invalid close happened before the fill
            ib = np.nonzero(inv_bar & (ends <= T[f]))[0]
            if len(ib):
                out[f"{en}_fill"] = 0; continue
            entry = float(L)
        out[f"{en}_fill"] = 1; out[f"{en}_entry"] = entry; out[f"{en}_fill_ms"] = int(T[f])
        sx = s_x[f:]; cmax = np.maximum.accumulate(sx); cmin = np.minimum.accumulate(sx)
        se = sgn * entry
        # first invalid close at/after fill
        bars_after = np.nonzero(inv_bar & (last >= f))[0]
        i_close = int(last[bars_after[0]] - f) if len(bars_after) else 10**9
        for sl in SLS:
            stop = (inv - sgn * 0.1 * w) if sl in ("VALH", "VALC") else (lo if bull else hi)
            if sl == "VALC":
                stop_hard = (lo if bull else hi) - sgn * 0.1 * w
            else:
                stop_hard = stop
            risk = se - sgn * stop
            if risk <= 0:
                continue
            out[f"{en}_{sl}_risk"] = risk
            i_s = int(np.searchsorted(-cmin, -sgn * stop_hard)) if True else 0  # first idx with cmin <= stop
            i_c = i_close if sl == "VALC" else 10**9
            for tp, (kind, k) in TPS.items():
                if kind == "R":
                    tgt = se + k * risk
                elif kind == "W":
                    tgt = sgn * edge + k * w
                    if tgt <= se:
                        continue
                else:
                    tgt = None
                i_t = int(np.searchsorted(cmax, tgt, side="left")) if tgt is not None else 10**9
                n = len(sx)
                i_s2 = i_s if i_s < n else 10**9
                i_t2 = i_t if i_t < n else 10**9
                first = min(i_s2, i_c, i_t2)
                if first == 10**9:
                    r, why = (sx[-1] - se) / risk, "time"
                elif first == i_s2:
                    r, why = (sx[i_s2] - se) / risk, "sl"
                elif first == i_c:
                    r, why = (sx[i_c] - se) / risk, "sl"
                else:
                    r, why = (tgt - se) / risk, "tp"
                out[f"{en}_{sl}_{tp}"] = float(r); out[f"{en}_{sl}_{tp}_why"] = why
    return out


def run(sym, y0, y1):
    d1 = pd.read_parquet(ROOT / f"data/processed/{sym}_D1.parquet"); d1["atr"] = atr(d1)
    d1_ct = pd.DatetimeIndex(d1["close_time"]).values
    h4 = pd.read_parquet(ROOT / f"data/processed/{sym}_H4.parquet")
    br = structure_breaks(h4, 2)
    br_t = pd.DatetimeIndex(br["time"]); br_d = br["direction"].astype(str).to_numpy()
    rows = []
    for day in pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D", tz=NY):
        if day.weekday() >= 5:
            continue
        t0 = ms((day - pd.Timedelta(hours=5)).tz_convert("UTC")); t1 = ms((day + pd.Timedelta(hours=16)).tz_convert("UTC"))
        tk = te.ticks(sym, t0, t1)
        if tk is None or len(tk[0]) < 100:
            continue
        for name, spec in RANGES.items():
            o = run_day(sym, *tk, day, name, spec)
            if o is None:
                continue
            if o["broke"]:
                ts = pd.Timestamp(o["break_ms"], unit="ms", tz="UTC")
                di = np.searchsorted(d1_ct, np.datetime64(ts.tz_localize(None), "ns"), side="right") - 1
                datr = float(d1.atr.values[di]) if di >= 0 else np.nan
                k = br_t.searchsorted(ts, side="right") - 1
                trend = (1 if br_d[k] == "bull" else -1) if k >= 0 else 0
                o["range_atr"] = o["w"] / datr if datr else np.nan
                o["trend_aligned"] = int(trend == (1 if o["direction"] == "bull" else -1))
            rows.append(o)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"ivb_{sym}_{y0}_{y1}.parquet")
    print(sym, y0, y1, len(d), "ranges", int(d.broke.sum()), "breaks", flush=True)


def load():
    return pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("ivb_*.parquet"))], ignore_index=True)


def add_net(d):
    pip = d.symbol.map(PIP)
    for en in ENTRIES:
        for sl in SLS:
            rc = f"{en}_{sl}_risk"
            if rc not in d:
                continue
            risk = d[rc]
            per_lot = np.where(d.symbol.str.endswith("USD"), risk * 1e5, risk * 1e5 / d[f"{en}_entry"])
            cost, slip = 5.0 / per_lot, 0.2 * pip / risk
            for tp in TPS:
                c = f"{en}_{sl}_{tp}"
                if c in d:
                    d[c + "_net"] = d[c] - cost - np.where(d[c + "_why"] == "sl", slip, 0)
    return d


def judge(x, col):
    x = x[x[col].notna()]
    if len(x) == 0:
        return dict(n=0, per_yr=0, win=np.nan, pf=np.nan, pf_21_23=np.nan, pf_24_26=np.nan, pos_years="0/0", pairs=[], PASS=False)
    y = x.when.dt.year
    yrs = x.groupby(y)[col].sum()
    pairs = [pf(q[col]) for _, q in x.groupby("symbol")]
    h1, h2 = pf(x.loc[y <= 2023, col]), pf(x.loc[y >= 2024, col])
    ok = (len(x) >= 150 and pf(x[col]) >= 1.10 and (yrs > 0).sum() >= 5 and sum(p > 1 for p in pairs) >= 2
          and h1 > 1 and h2 > 1)
    return dict(n=len(x), per_yr=round(len(x) / 5.75), win=round(100 * float((x[col] > 0).mean()), 1), pf=pf(x[col]),
                pf_21_23=h1, pf_24_26=h2, pos_years=f"{int((yrs > 0).sum())}/{yrs.size}", pairs=pairs, PASS=ok)


def report():
    d = load()
    d = add_net(d[d.broke == 1].copy())
    d["when"] = pd.to_datetime(d.break_ms, unit="ms")
    learn_all, test_all = d[d.when.dt.year <= 2020], d[d.when.dt.year >= 2021]
    # statistical TP per pair+range: W level nearest to the 65%-reached excursion (2016-20 only)
    stat = {}
    for (s, rg), g in learn_all.groupby(["symbol", "range"]):
        q = g.mfe_w.quantile(0.35)
        stat[(s, rg)] = min(("W05", 0.5), ("W1", 1.0), ("W15", 1.5), ("W2", 2.0), key=lambda kv: abs(kv[1] - q))[0]
    print("statistical TP (65% reached, 2016-20):", stat)
    for en in ENTRIES:
        for sl in SLS:
            c = f"{en}_{sl}_STAT_net"
            d[c] = np.nan
            for (s, rg), tp in stat.items():
                src = f"{en}_{sl}_{tp}_net"
                if src in d:
                    m = (d.symbol == s) & (d.range == rg)
                    d.loc[m, c] = d.loc[m, src]
    learn, test = d[d.when.dt.year <= 2020], d[d.when.dt.year >= 2021]
    rows = []
    for rg in RANGES:
        L, T = learn[learn.range == rg], test[test.range == rg]
        for en in ENTRIES:
            for sl in SLS:
                for tp in list(TPS) + ["STAT"]:
                    c = f"{en}_{sl}_{tp}_net"
                    if c not in d:
                        continue
                    lx = L[c].dropna()
                    rows.append(dict(range=rg, entry=en, sl=sl, tp=tp, learn_n=len(lx), learn_pf=pf(lx), **judge(T, c)))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "data/research/orb_ivb_all.csv", index=False)
    best = (t[t.learn_n >= 150].sort_values("learn_pf", ascending=False).groupby("range").head(1))
    best.to_csv(ROOT / "data/research/orb_ivb_best.csv", index=False)
    return t, best


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        t, best = report()
        print("\nBEST COMBO PER RANGE (chosen on 2016-20 only):")
        print(best.to_string(index=False))
