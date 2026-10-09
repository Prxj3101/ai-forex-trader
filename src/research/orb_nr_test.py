r"""
ORB idea 1 + 2 (9 Oct 2026) on REAL TICKS. Rules fixed BEFORE running.

Ranges (NY time): ASIA 19:00-02:00 / LON30 03:00-03:30 / LON60 03:00-04:00, trade window to 11:00, EOD 16:00.
BRK (base ORB): first tick breaking the range (BUY: bid > hi -> fill at ask), SL other side of range, TP 1R / 2R / EOD.

IDEA 1 - COMPRESSION (Toby Crabel), only info known before the range ends:
  NR4 / NR7  = yesterday's daily range (last closed D1) is the smallest of the last 4 / 7 days
  INSIDE     = yesterday was an inside day
  TIGHT      = today's range / average of the previous 20 same ranges <= 30th pct of 2016-20 (quantile from learn only)
IDEA 2 - FAILED BREAKOUT (fakeout) reversal, trade in the OPPOSITE direction of the first break:
  FAKE = after the break, first M5 close back inside the range (before 11:00) -> enter at next tick;
         SL = extreme reached after the break + 0.1 x range; TP 1R / 2R / OPP (other range side) / EOD
  SAR  = after the break, price breaks the OTHER side of the range (before 11:00) -> enter there; SL = range far side
Costs $5/lot + 0.2 pip on stop exits.
PASS (2021-26): net PF >= 1.10, >= 5/6 years positive, >= 2/3 pairs > 1, >= 150 trades, PF > 1 in 2021-23 and 2024-26,
  AND learn (2016-20) PF >= 1.0 for the same rule (no rule that only works in the test years).
Chalana: python src\research\orb_nr_test.py run EURUSD 2016 2026 | python src\research\orb_nr_test.py report
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402

OUT = ROOT / "data/research/orb_nr"
NY = "America/New_York"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
RANGES = {"ASIA": ((19, 0), (2, 0), (11, 0)), "LON30": ((3, 0), (3, 30), (11, 0)), "LON60": ((3, 0), (4, 0), (11, 0))}
EOD = (16, 0)
BAR = 300_000
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan  # noqa: E731


def ms(ts):
    return int(ts.value // 1_000_000)


def sim(T, A, B, bull, f, entry, stop, tgts):
    """Trade from tick f. tgts = {name: price or None(EOD)}. Returns {name: (r, why)} and risk."""
    sgn = 1 if bull else -1
    sx = sgn * (B[f:] if bull else A[f:])
    se, ss = sgn * entry, sgn * stop
    risk = se - ss
    if risk <= 0 or len(sx) == 0:
        return None, risk
    cmax = np.maximum.accumulate(sx); cmin = np.minimum.accumulate(sx)
    i_s = int(np.searchsorted(-cmin, -ss)); n = len(sx)
    out = {}
    for name, tg in tgts.items():
        i_t = int(np.searchsorted(cmax, sgn * tg)) if tg is not None else n
        if tg is not None and sgn * tg <= se:
            continue
        if i_s < n and i_s <= i_t:
            out[name] = ((sx[i_s] - se) / risk, "sl")
        elif i_t < n:
            out[name] = ((sgn * tg - se) / risk, "tp")
        else:
            out[name] = ((sx[-1] - se) / risk, "time")
    return out, risk


def run_day(sym, t, a, b, day, name, spec):
    (h0, m0), (h1, m1), (h2, m2) = spec
    r_start = (day - pd.Timedelta(days=1) if h0 > h1 else day) + pd.Timedelta(hours=h0, minutes=m0)
    r_end = ms((day + pd.Timedelta(hours=h1, minutes=m1)).tz_convert("UTC"))
    w_end = ms((day + pd.Timedelta(hours=h2, minutes=m2)).tz_convert("UTC"))
    eod = ms((day + pd.Timedelta(hours=EOD[0], minutes=EOD[1])).tz_convert("UTC"))
    i0, i1, i2 = np.searchsorted(t, [ms(r_start.tz_convert("UTC")), r_end, eod])
    if i1 - i0 < 50 or i2 - i1 < 50:
        return None
    hi, lo = float(b[i0:i1].max()), float(b[i0:i1].min()); w = hi - lo
    if w <= 0:
        return None
    T, A, B = t[i1:i2], a[i1:i2], b[i1:i2]
    out = dict(symbol=sym, day=day.date(), range=name, hi=hi, lo=lo, w=w, range_end_ms=r_end, broke=0)
    win = T < w_end
    up = np.nonzero(win & (B > hi))[0]; dn = np.nonzero(win & (B < lo))[0]
    if not len(up) and not len(dn):
        return out
    bull = (up[0] if len(up) else 10**9) < (dn[0] if len(dn) else 10**9)
    i = up[0] if bull else dn[0]
    entry = float(A[i] if bull else B[i])
    out.update(broke=1, direction="bull" if bull else "bear", break_ms=int(T[i]), entry=entry)
    stop = lo if bull else hi
    res, risk = sim(T, A, B, bull, i, entry, stop, {"1R": None if False else entry + (1 if bull else -1) * abs(entry - stop),
                                                     "2R": entry + (2 if bull else -2) * abs(entry - stop), "EOD": None})
    if res:
        out["BRK_risk"] = risk; out["BRK_entry"] = entry
        for k, (r, why) in res.items():
            out[f"BRK_{k}"], out[f"BRK_{k}_why"] = r, why
    # ---- FAKE: first M5 close back inside after the break
    k = (T - r_end) // BAR
    last = np.nonzero(np.diff(k, append=k[-1] + 1))[0]
    ends = r_end + (k[last] + 1) * BAR
    cb = B[last]
    back = (ends > T[i]) & (ends <= w_end) & ((cb < hi) if bull else (cb > lo))
    bi = np.nonzero(back)[0]
    if len(bi):
        j = int(last[bi[0]]) + 1
        if j < len(T):
            rbull = not bull
            fe = float(A[j] if rbull else B[j])
            ext = float(B[i:j].max()) if bull else float(A[i:j].min())
            fstop = ext + 0.1 * w if bull else ext - 0.1 * w
            rr = abs(fe - fstop); sg = 1 if rbull else -1
            res, risk = sim(T, A, B, rbull, j, fe, fstop, {"1R": fe + sg * rr, "2R": fe + 2 * sg * rr,
                                                          "OPP": lo if bull else hi, "EOD": None})
            if res:
                out.update(FAKE_risk=risk, FAKE_entry=fe, FAKE_ms=int(T[j]))
                for kk, (r, why) in res.items():
                    out[f"FAKE_{kk}"], out[f"FAKE_{kk}_why"] = r, why
    # ---- SAR: opposite side broken after the first break
    opp = np.nonzero(win & (np.arange(len(T)) > i) & ((B < lo) if bull else (B > hi)))[0]
    if len(opp):
        j = int(opp[0]); rbull = not bull
        se_ = float(A[j] if rbull else B[j]); sstop = hi if bull else lo
        rr = abs(se_ - sstop); sg = 1 if rbull else -1
        res, risk = sim(T, A, B, rbull, j, se_, sstop, {"1R": se_ + sg * rr, "2R": se_ + 2 * sg * rr, "EOD": None})
        if res:
            out.update(SAR_risk=risk, SAR_entry=se_, SAR_ms=int(T[j]))
            for kk, (r, why) in res.items():
                out[f"SAR_{kk}"], out[f"SAR_{kk}_why"] = r, why
    return out


def run(sym, y0, y1):
    d1 = pd.read_parquet(ROOT / f"data/processed/{sym}_D1.parquet")
    rng = (d1.high - d1.low).to_numpy(); ct = pd.DatetimeIndex(d1.close_time).values
    hi_, lo_ = d1.high.to_numpy(), d1.low.to_numpy()
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
            di = np.searchsorted(ct, np.datetime64(pd.Timestamp(o["range_end_ms"], unit="ms"), "ns"), side="right") - 1
            if di >= 7:
                o["nr4"] = int(rng[di] <= rng[di - 3:di + 1].min())
                o["nr7"] = int(rng[di] <= rng[di - 6:di + 1].min())
                o["inside"] = int(hi_[di] <= hi_[di - 1] and lo_[di] >= lo_[di - 1])
            rows.append(o)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"nr_{sym}_{y0}_{y1}.parquet")
    print(sym, y0, y1, len(d), "ranges", int(d.broke.sum()), "breaks",
          int(d.get("FAKE_entry", pd.Series(dtype=float)).notna().sum()), "fakes",
          int(d.get("SAR_entry", pd.Series(dtype=float)).notna().sum()), "sar", flush=True)


def load():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("nr_*.parquet"))], ignore_index=True)
    d = d.sort_values(["symbol", "range", "day"]).reset_index(drop=True)
    d["w_ratio"] = d.w / d.groupby(["symbol", "range"]).w.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    pip = d.symbol.map(PIP)
    for st, tps in (("BRK", ("1R", "2R", "EOD")), ("FAKE", ("1R", "2R", "OPP", "EOD")), ("SAR", ("1R", "2R", "EOD"))):
        risk = d[f"{st}_risk"]
        per_lot = np.where(d.symbol.str.endswith("USD"), risk * 1e5, risk * 1e5 / d[f"{st}_entry"])
        cost, slip = 5.0 / per_lot, 0.2 * pip / risk
        for tp in tps:
            c = f"{st}_{tp}"
            d[c + "_net"] = d[c] - cost - np.where(d[c + "_why"] == "sl", slip, 0)
    d["when"] = pd.to_datetime(d.day)
    return d


def judge(x, col, learn_pf):
    x = x[x[col].notna()]
    if len(x) == 0:
        return dict(n=0, per_yr=0, win=np.nan, pf=np.nan, pf_21_23=np.nan, pf_24_26=np.nan, pos_years="0/0", pairs=[], PASS=False)
    y = x.when.dt.year
    yrs = x.groupby(y)[col].sum()
    pairs = [pf(q[col]) for _, q in x.groupby("symbol")]
    h1, h2 = pf(x.loc[y <= 2023, col]), pf(x.loc[y >= 2024, col])
    ok = (len(x) >= 150 and pf(x[col]) >= 1.10 and (yrs > 0).sum() >= 5 and sum(p > 1 for p in pairs) >= 2
          and h1 > 1 and h2 > 1 and learn_pf >= 1.0)
    return dict(n=len(x), per_yr=round(len(x) / 5.75), win=round(100 * float((x[col] > 0).mean()), 1), pf=pf(x[col]),
                pf_21_23=h1, pf_24_26=h2, pos_years=f"{int((yrs > 0).sum())}/{yrs.size}", pairs=pairs, PASS=ok)


def report():
    d = load()
    learn_all = d[d.when.dt.year <= 2020]
    rows = []
    for rg in RANGES:
        g = d[d.range == rg]
        q30 = learn_all[learn_all.range == rg].w_ratio.quantile(0.3)
        filters = {"(none)": g.broke == 1, "NR4": g.nr4 == 1, "NR7": g.nr7 == 1, "INSIDE": g.inside == 1,
                   "TIGHT": g.w_ratio <= q30, "NR4 or TIGHT": (g.nr4 == 1) | (g.w_ratio <= q30),
                   "NOT tight (wide)": g.w_ratio > q30}
        for st, tps in (("BRK", ("1R", "2R", "EOD")), ("FAKE", ("1R", "2R", "OPP", "EOD")), ("SAR", ("1R", "2R", "EOD"))):
            for tp in tps:
                c = f"{st}_{tp}_net"
                for fn, m in filters.items():
                    x = g[m.fillna(False)]
                    L, T = x[x.when.dt.year <= 2020], x[x.when.dt.year >= 2021]
                    lp = pf(L[c].dropna())
                    rows.append(dict(range=rg, setup=st, tp=tp, filter=fn, learn_n=int(L[c].notna().sum()), learn_pf=lp,
                                     **judge(T, c, lp if lp == lp else 0)))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "data/research/orb_nr_all.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        t = report()
        print(t.to_string(index=False))
