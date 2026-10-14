r"""
GOLD "FIB 7" (Prajwal's strategy, 14 Oct 2026) on REAL TICKS. Rules fixed BEFORE running (from his chat + screenshots).

MOVE    15m and 1h charts separately (bid bars). Swing high = bar high above the 3 bars before and >= the 3 bars after
        (swing low mirror); a swing is only KNOWN when the 3rd bar after it has closed (no look-ahead).
        Swings alternate (two highs in a row -> keep the higher). Up move = swing low L -> next swing high H,
        size H-L >= 1.0 x ATR(14) of that chart. Down move = mirror (SELL setups).
FIB     1 = L, 0 = H. ENTRY = buy LIMIT at the 0.705 level. Zone = 0.705 .. 0.786.
        Created when H is confirmed. Skipped if price is already at/below 0.705 at that moment (missed).
VALID   until filled (one trade per zone), or SL level touched before the fill, or 10 days old.
        (He said: old zones stay important levels; no other expiry.)
SL      A = 0.816 (just below 0.786, like his screenshot)  B = 0.886  C = 1.0 (move start)
TP      1R | 1.5R | 2R | FIB0 (back to the move's high/low). Max hold 3 days (tick_engine).
FILL    first tick with ask <= entry (BUY) / bid >= entry (SELL); exits on ticks (tick_engine.run_exit, same as all other tests).
COSTS   $7 per lot round trip (100 oz) + $0.05 slippage on stop exits. Spread = real Dukascopy bid/ask (conservative).
STRONG SIGNS (measured, not required):
  BOS     the move broke the previous swing high (up move) / previous swing low (down move)
  SESSION L and H are the low and high of ONE session (NY time: Asia 19-02, London 02-08, NY 08-16) and the move is inside it
  SWEEP   a previous session's low (BUY) / high (SELL) of the last 3 finished sessions sits between 0.705 and 0.505,
          so the pullback sweeps that liquidity on its way into the zone
PASS (fixed now): PF >= 1.10 in 2016-2020 AND >= 1.10 in 2021-2026, >= 50 trades/year, and neighbour variants (other SL/TP)
          not far off - no single lucky combo.
V2 (14 Oct, from his Fib 7 Check answers - his NO reasons; rules fixed BEFORE running):
  BIG SWINGS  ATR zigzag: a swing high/low is confirmed when price moves 1.5 x ATR(14) away from it (bar close known);
              leg L -> H must be >= 3 x ATR (full big move, no inner legs).
  FRESH ZONE  price must NOT have touched the 0.705 level between the move end and the zone's confirmation
              (after that the first touch is the fill) - "zone already used / breached = invalid".
  TREND       BUY only if the last closed H4 bar closed above H4 EMA(50); SELL only below.
  Everything else as v1 (entry 0.705, SL A/B/C, TP 1R/1.5R/2R/FIB0, 10 days, ticks, $7 + $0.05 slip).
  PASS (fixed): PF >= 1.10 in 2016-20 AND in 2021-26, >= 30 trades/year, neighbours (other SL/TP) not far off.
Chalana:  python src\research\gold_fib.py run [v2]       (every year that has converted gold ticks; resumable)
          python src\research\gold_fib.py report [v2]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402

SYM = "XAUUSD"
OUT = ROOT / "data/research/gold_fib"
OUT2 = ROOT / "data/research/gold_fib_v2"
NY = "America/New_York"
COMM, SLIP, OZ = 7.0, 0.05, 100.0
N_SIDE = 3
MIN_ATR = 1.0
VALID_DAYS = 10
SLS = {"A_0.816": 0.816, "B_0.886": 0.886, "C_1.0": 1.0}
TPS = {"1R": 1.0, "1.5R": 1.5, "2R": 2.0, "FIB0": None}
SESS = [("ASIA", 19, 2), ("LONDON", 2, 8), ("NY", 8, 16)]


def bars(tf):
    b = pd.read_parquet(ROOT / f"data/processed/{SYM}_{tf}.parquet", columns=["open", "high", "low", "close", "close_time"])
    pc = b.close.shift()
    b["atr"] = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs())).rolling(14).mean()
    return b


def swings(b):
    """Alternating swings: list of (kind 'H'/'L', bar index, price, confirm_time)."""
    h, l = b.high.to_numpy(), b.low.to_numpy()
    n = len(b)
    raw = []
    for i in range(N_SIDE, n - N_SIDE):
        if h[i] > h[i - N_SIDE:i].max() and h[i] >= h[i + 1:i + 1 + N_SIDE].max():
            raw.append(("H", i, h[i]))
        if l[i] < l[i - N_SIDE:i].min() and l[i] <= l[i + 1:i + 1 + N_SIDE].min():
            raw.append(("L", i, l[i]))
    out = []
    for k, i, p in raw:
        if out and out[-1][0] == k:
            if (k == "H" and p > out[-1][2]) or (k == "L" and p < out[-1][2]):
                out[-1] = (k, i, p)
            continue
        out.append((k, i, p))
    ct = b.close_time.dt.tz_convert(None).to_numpy()            # naive UTC
    return [(k, i, p, ct[i + N_SIDE]) for k, i, p in out]


def session_of(ts):
    h = ts.tz_convert(NY).hour
    for name, a, z in SESS:
        if (a <= h < z) if a < z else (h >= a or h < z):
            return name
    return None


def session_table(m5):
    """Every finished session: (end time, name, low, high)."""
    ny = m5.index.tz_convert(NY)
    hr = ny.hour
    lab = np.full(len(m5), "", dtype=object)
    for name, a, z in SESS:
        mask = ((hr >= a) & (hr < z)) if a < z else ((hr >= a) | (hr < z))
        lab[mask] = name
    d = pd.DataFrame({"lab": lab, "high": m5.high.to_numpy(), "low": m5.low.to_numpy(),
                      "ct": m5.close_time.dt.tz_convert(None).to_numpy()})          # naive UTC
    d["grp"] = (d.lab != d.lab.shift()).cumsum()
    g = d[d.lab != ""].groupby("grp").agg(name=("lab", "first"), lo=("low", "min"), hi=("high", "max"), end=("ct", "max"))
    return g.reset_index(drop=True)


def zones(tf, b, sess):
    sw = swings(b)
    ct = b.close_time.dt.tz_convert(None).to_numpy()
    cl = b.close.to_numpy()
    out = []
    for j in range(2, len(sw)):
        k1, i1, p1, _ = sw[j - 1]
        k2, i2, p2, conf = sw[j]
        prev_same = sw[j - 2]                                   # the swing of the same kind before p2
        size = abs(p2 - p1)
        atr = b.atr.iloc[i2]
        if not np.isfinite(atr) or size < MIN_ATR * atr:
            continue
        bull = k1 == "L" and k2 == "H"                           # up move -> BUY the pullback
        H, L = (p2, p1) if bull else (p1, p2)
        lvl = lambda f: H - f * (H - L) if bull else L + f * (H - L)  # noqa: E731
        entry = lvl(0.705)
        ci = int(np.searchsorted(ct, conf))                      # bar whose close confirmed the swing
        now = cl[min(ci, len(cl) - 1)]
        if (bull and now <= entry) or (not bull and now >= entry):
            continue                                             # pullback already there -> missed
        t1, t2 = b.index[i1], b.index[i2]
        bos = bool(p2 > prev_same[2]) if bull else bool(p2 < prev_same[2])
        s1, s2 = session_of(t1), session_of(t2)
        session = False
        if s1 is not None and s1 == s2 and (t2 - t1) < pd.Timedelta(hours=10):
            m = sess[(sess.end > np.datetime64(t1.tz_convert(None))) & (sess.name == s1)].head(1)
            if len(m):
                session = bool(np.isclose(m.lo.iloc[0], L) and np.isclose(m.hi.iloc[0], H))
        fin = sess[sess.end <= conf].tail(3)
        lvls = fin.lo.to_numpy() if bull else fin.hi.to_numpy()
        a5 = lvl(0.505)
        sweep = bool(np.any((lvls >= min(entry, a5)) & (lvls <= max(entry, a5))))
        out.append(dict(tf=tf, bull=bull, t_start=t1, t_end=t2, created=pd.Timestamp(conf).tz_localize("UTC"),
                        H=H, L=L, size=size, atr=atr, entry=entry, **{f"sl_{k}": lvl(f) for k, f in SLS.items()},
                        bos=bos, session=session, sweep=sweep))
    return pd.DataFrame(out)


def zigzag(b, k=1.5):
    """ATR zigzag: swings confirmed when price moves k x ATR away. Returns (kind, bar idx, price, confirm close_time)."""
    h, l, a = b.high.to_numpy(), b.low.to_numpy(), b.atr.to_numpy()
    ct = b.close_time.dt.tz_convert(None).to_numpy()
    out, d, ei, ep = [], 0, 0, h[0]
    for i in range(1, len(b)):
        if not np.isfinite(a[i]):
            continue
        if d >= 0:                                              # looking for a high (or no direction yet)
            if h[i] > ep or d == 0 and h[i] > ep:
                ei, ep = i, h[i]
            if d == 0 and l[i] < ep - k * a[i]:
                d = 1
            if d == 1 and l[i] <= ep - k * a[i]:
                out.append(("H", ei, ep, ct[i])); d, ei, ep = -1, i, l[i]
                continue
        if d == -1:
            if l[i] < ep:
                ei, ep = i, l[i]
            if h[i] >= ep + k * a[i]:
                out.append(("L", ei, ep, ct[i])); d, ei, ep = 1, i, h[i]
    return out


def zones_v2(tf, b, sess, h4):
    sw = zigzag(b)
    ct = b.close_time.dt.tz_convert(None).to_numpy()
    hi, lo, cl = b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
    h4ct = h4.close_time.dt.tz_convert(None).to_numpy()
    h4c, h4e = h4.close.to_numpy(), h4.close.ewm(span=50, adjust=False).mean().to_numpy()
    out = []
    for j in range(2, len(sw)):
        k1, i1, p1, _ = sw[j - 1]
        k2, i2, p2, conf = sw[j]
        size = abs(p2 - p1)
        atr = b.atr.iloc[i2]
        if not np.isfinite(atr) or size < 3.0 * atr:
            continue
        bull = k1 == "L" and k2 == "H"
        H, L = (p2, p1) if bull else (p1, p2)
        lvl = lambda f: H - f * (H - L) if bull else L + f * (H - L)  # noqa: E731
        entry = lvl(0.705)
        ci = int(np.searchsorted(ct, conf))
        seg = slice(i2 + 1, min(ci, len(cl) - 1) + 1)            # bars after the move end up to confirmation
        if (bull and (lo[seg].min() if seg.start < seg.stop else np.inf) <= entry) or \
           (not bull and (hi[seg].max() if seg.start < seg.stop else -np.inf) >= entry):
            continue                                             # zone already touched -> not fresh
        if (bull and cl[min(ci, len(cl) - 1)] <= entry) or (not bull and cl[min(ci, len(cl) - 1)] >= entry):
            continue
        k4 = int(np.searchsorted(h4ct, conf, side="right")) - 1   # last H4 bar closed by the confirmation
        if k4 < 60 or (bull and h4c[k4] <= h4e[k4]) or (not bull and h4c[k4] >= h4e[k4]):
            continue                                             # against the H4 trend
        t1, t2 = b.index[i1], b.index[i2]
        prev_same = sw[j - 2]
        bos = bool(p2 > prev_same[2]) if bull else bool(p2 < prev_same[2])
        s1, s2 = session_of(t1), session_of(t2)
        session = False
        if s1 is not None and s1 == s2 and (t2 - t1) < pd.Timedelta(hours=10):
            m = sess[(sess.end > np.datetime64(t1.tz_convert(None))) & (sess.name == s1)].head(1)
            if len(m):
                session = bool(np.isclose(m.lo.iloc[0], L) and np.isclose(m.hi.iloc[0], H))
        fin = sess[sess.end <= conf].tail(3)
        lv = fin.lo.to_numpy() if bull else fin.hi.to_numpy()
        a5 = lvl(0.505)
        sweep = bool(np.any((lv >= min(entry, a5)) & (lv <= max(entry, a5))))
        out.append(dict(tf=tf, bull=bull, t_start=t1, t_end=t2, created=pd.Timestamp(conf).tz_localize("UTC"),
                        H=H, L=L, size=size, atr=atr, entry=entry, **{f"sl_{k}": lvl(f) for k, f in SLS.items()},
                        bos=bos, session=session, sweep=sweep))
    return pd.DataFrame(out)


def trade(z):
    """Fill + all SL x TP exits for one zone (ticks)."""
    bull = z.bull
    c_ms = int(z.created.value // 10**6)
    end_ms = c_ms + VALID_DAYS * 86_400_000
    # cancel: SL A (tightest) touched before the fill -> zone dead for every variant (same entry)
    f = te.find_fill(SYM, bull, z.entry, c_ms, end_ms)
    if f in (None, "missing"):
        return dict(status="no_fill" if f is None else "missing")
    rows = dict(status="filled", fill_ms=f)
    for sk in SLS:
        sl = z[f"sl_{sk}"]
        risk = abs(z.entry - sl)
        for tk, tr in TPS.items():
            tpr = tr if tr is not None else abs(z.H - z.entry if bull else z.entry - z.L) / risk
            res = te.run_exit(SYM, bull, z.entry, sl, f, "tp", tpr)
            net = res["r"] - COMM / (risk * OZ) - (SLIP / risk if res["reason"] in ("sl", "trail", "be") else 0.0)
            rows[f"r_{sk}_{tk}"] = net
    rows["risk_A"] = abs(z.entry - z["sl_A_0.816"])
    return rows


def run(v2=False):
    out_dir = OUT2 if v2 else OUT
    store = ROOT / "data/ticks/store" / SYM
    years = sorted(int(p.name[6:]) for p in store.glob("_done_*"))
    m5 = pd.read_parquet(ROOT / f"data/processed/{SYM}_M5.parquet", columns=["high", "low", "close_time"])
    sess = session_table(m5)
    out_dir.mkdir(parents=True, exist_ok=True)
    h4 = bars("H4")
    for tf in ["M15", "H1"]:
        b = bars(tf)
        allz = zones_v2(tf, b, sess, h4) if v2 else zones(tf, b, sess)
        for y in years:
            f = out_dir / f"{tf}_{y}.parquet"
            if f.exists():
                print(tf, y, "already done"); continue
            zz = allz[allz.created.dt.year == y].reset_index(drop=True)
            t0, rows = time.time(), []
            for _, z in zz.iterrows():
                rows.append({**z.to_dict(), **trade(z)})
            d = pd.DataFrame(rows)
            d.to_parquet(f)
            print(f"{tf} {y}: zones {len(zz)}, filled {int((d.status == 'filled').sum())}, {time.time() - t0:.0f}s", flush=True)


def pf(r):
    r = r.dropna()
    return round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan


def report(v2=False):
    d = pd.concat([pd.read_parquet(f) for f in sorted((OUT2 if v2 else OUT).glob("*_20*.parquet"))], ignore_index=True)
    d = d[d.status == "filled"].copy()
    d["year"] = pd.to_datetime(d.fill_ms, unit="ms").dt.year
    yrs = d.year.nunique()
    print(f"years with ticks: {sorted(d.year.unique())}")
    rows = []
    for tf, g in d.groupby("tf"):
        for sk in SLS:
            for tk in TPS:
                c = f"r_{sk}_{tk}"
                L, T = g[g.year <= 2020][c], g[g.year >= 2021][c]
                rows.append(dict(tf=tf, sl=sk, tp=tk, trades_yr=round(len(g) / yrs), win=round(float((g[c] > 0).mean()) * 100),
                                 pf_all=pf(g[c]), pf_2016_20=pf(L), pf_2021_26=pf(T), R=round(g[c].sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSTRONG SIGNS (SL A, TP 1R) - PF with the sign vs without:")
    for tf, g in d.groupby("tf"):
        c = "r_A_0.816_1R"
        for s in ["bos", "session", "sweep"]:
            print(f"  {tf} {s:8s} with: PF {pf(g[g[s]][c])} (n {int(g[s].sum())}) | without: PF {pf(g[~g[s]][c])} (n {int((~g[s]).sum())})")
        allin = g.bos & g.sweep
        print(f"  {tf} bos+sweep: PF {pf(g[allin][c])} (n {int(allin.sum())})")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]](v2=len(sys.argv) > 2 and sys.argv[2] == "v2")
