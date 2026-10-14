r"""
GOLD FIB 7 - CONFLUENCE + 5m CONFIRMATION (15 Oct 2026). Two ideas from HIS rules that were never tested:
 (1) CONFLUENCE: "old zones stay important levels". A zone counts as confluence when its 0.705-0.786 band
     (+/- 0.25 ATR of its chart) overlaps, at zone creation time (known then):
       OLD  = the 0.705-0.786 band of an earlier zone (M15 K2.5 or H1 K2.5, same direction, created in the last 20 days)
       DAY  = previous day's high or low
       SESS = high or low of one of the last 3 finished sessions (Asia / London / NY)
 (2) 5m CONFIRMATION instead of the blind limit: after the FIRST touch of 0.705, wait (max 12 M5 bars = 1 h) for an
     M5 bar that CLOSES beyond the high (BUY) / low (SELL) of the 3 bars before it (micro BOS). Enter at the next tick
     (real ask/bid). Cancelled if price reaches fib 1.0 first, or no trigger within 1 h.
     SL: SW = beyond the extreme made since the touch + 0.05 of the move | S100 = fib 1.0. TP 1R / 1.5R / 2R.
 BASE (for comparison): limit at 0.705, first touch, SL 0.886 / 1.0, same TPs. Same zones + dead rules as the grid
 (fib 0 broken before entry / M5 pre-touch -> dead), 10 days. Costs $7 RT + $0.05 slip + Dukascopy spread, 3-day hold.
 Sets: M15 K2.5, H1 K2.5, H1 K3.5. Judge on 2016-20 first; 2021-26 shown too (that period was already looked at in
 earlier Fib 7 tests - so a pass there is weaker evidence than a fresh holdout).
Chalana: python src\research\gold_fib_conf.py run | report
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.research import gold_fib as gf  # noqa: E402
from src.research.gold_fib_big import m5_arrays  # noqa: E402

OUT = ROOT / "data/research/gold_fib_conf"
SETS = [("M15", 2.5), ("H1", 2.5), ("H1", 3.5)]
TPS = {"1R": 1.0, "1.5R": 1.5, "2R": 2.0}
COMM, SLIP, OZ, DAY = 7.0, 0.05, 100.0, 86_400_000


def _ms(x):
    return pd.DatetimeIndex(x).tz_convert(None).to_numpy().astype("datetime64[ms]").astype("int64")


def exits(out, pre, tt, aa, bb, bull, px, sl):
    risk = (px - sl) if bull else (sl - px)
    if risk <= 0:
        return
    for k, tr in TPS.items():
        res = te.exit_arrays(tt, aa, bb, bull, px, sl, "tp", tr)
        out[f"{pre}_{k}"] = res["r"] - COMM / (risk * OZ) - (SLIP / risk if res["reason"] == "sl" else 0.0)


def hold(t, a, b, i):
    j = np.searchsorted(t, t[i] + te.HOLD_MS)
    if t[i] + te.HOLD_MS > t[-1]:
        return te.ticks("XAUUSD", int(t[i]), int(t[i]) + te.HOLD_MS)
    return t[i:j], a[i:j], b[i:j]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    hi, lo, op, ct = m5_arrays()
    m5c = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["close"]).close.to_numpy()
    sess = gf.session_table(pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"]))
    s_end = sess.end.to_numpy().astype("datetime64[ms]").astype("int64")
    d1 = gf.bars("D1")
    d1ct = _ms(d1.close_time)
    oldz = pd.concat([pd.read_parquet(ROOT / f"data/research/gold_fib_ml/{tf}_K2.5_zones_all.parquet") for tf in ["M15", "H1"]])
    oldz["c_ms"] = _ms(oldz.created)
    oldz["b_hi"] = np.where(oldz.bull, oldz.H - 0.705 * (oldz.H - oldz.L), oldz.L + 0.786 * (oldz.H - oldz.L))
    oldz["b_lo"] = np.where(oldz.bull, oldz.H - 0.786 * (oldz.H - oldz.L), oldz.L + 0.705 * (oldz.H - oldz.L))
    oldz = oldz.sort_values("c_ms")
    oc, ob, ohi, olo = oldz.c_ms.to_numpy(), oldz.bull.to_numpy(), oldz.b_hi.to_numpy(), oldz.b_lo.to_numpy()
    t_start = time.time()
    for tf, k in SETS:
        z = pd.read_parquet(ROOT / f"data/research/gold_fib_ml/{tf}_K{k}_zones_all.parquet")
        for y in sorted(z.created.dt.year.unique()):
            f = OUT / f"{tf}_K{k}_{y}.parquet"
            if f.exists():
                continue
            t0, rows = time.time(), []
            for r in z[z.created.dt.year == y].to_dict("records"):
                bull, H, L = bool(r["bull"]), r["H"], r["L"]
                lvl = lambda f_: H - f_ * (H - L) if bull else L + f_ * (H - L)  # noqa: E731
                e, f1 = lvl(0.705), lvl(1.0)
                c_ms = int(r["created"].value // 10**6)
                i0 = np.searchsorted(op, int(r["t_end"].timestamp() * 1000), side="right")
                i1 = np.searchsorted(ct, c_ms, side="right")
                if ((lo[i0:i1] <= e).any() or (hi[i0:i1] > H).any()) if bull else ((hi[i0:i1] >= e).any() or (lo[i0:i1] < L).any()):
                    continue
                tk = te.ticks("XAUUSD", c_ms, c_ms + 10 * DAY)
                if tk is None:
                    continue
                t, a, b = tk
                hit = np.nonzero(b <= e)[0] if bull else np.nonzero(b >= e)[0]
                if not len(hit):
                    continue
                i = hit[0]
                if ((b[:i + 1] > H).any() if bull else (b[:i + 1] < L).any()):
                    continue
                # confluence (known at creation)
                tol = 0.25 * r["atr"]
                band_lo, band_hi = min(lvl(0.705), lvl(0.786)) - tol, max(lvl(0.705), lvl(0.786)) + tol
                m = (oc < c_ms) & (oc >= c_ms - 20 * DAY) & (ob == bull) & (ohi >= band_lo) & (olo <= band_hi)
                m &= ~((np.abs(ohi - max(lvl(0.705), lvl(0.786))) < 1e-6))                  # not itself
                kd = np.searchsorted(d1ct, c_ms, side="right") - 1
                dl = [d1.high.iloc[kd], d1.low.iloc[kd]]
                ks = np.searchsorted(s_end, c_ms, side="right")
                sl3 = sess.iloc[max(ks - 3, 0):ks]
                slv = list(sl3.hi) + list(sl3.lo)
                inb = lambda v: band_lo <= v <= band_hi  # noqa: E731
                o = {c: r[c] for c in ["tf", "bull", "created", "H", "L", "size", "atr"]}
                o.update(conf_old=bool(m.any()), conf_day=any(inb(v) for v in dl), conf_sess=any(inb(v) for v in slv))
                # BASE: limit at 0.705, first touch
                px = float(a[i]) if bull else float(b[i])
                tt, aa, bb = hold(t, a, b, i)
                exits(o, "r_lim_S886", tt, aa, bb, bull, px, lvl(0.886))
                exits(o, "r_lim_S100", tt, aa, bb, bull, px, f1)
                # 5m CONFIRMATION
                j0 = np.searchsorted(ct, int(t[i]), side="right")          # first M5 bar closing after the touch
                ext = (lo if bull else hi)[j0]
                for j in range(j0, min(j0 + 12, len(ct))):
                    ext = min(ext, lo[j]) if bull else max(ext, hi[j])
                    if (ext <= f1) if bull else (ext >= f1):
                        break
                    if j - 3 < 0:
                        continue
                    ref = hi[j - 3:j].max() if bull else lo[j - 3:j].min()
                    if (m5c[j] > ref) if bull else (m5c[j] < ref):
                        ie = np.searchsorted(t, ct[j])
                        if ie >= len(t):
                            break
                        pxc = float(a[ie]) if bull else float(b[ie])
                        tt, aa, bb = hold(t, a, b, ie)
                        buf = 0.05 * (H - L)
                        exits(o, "r_cf_SW", tt, aa, bb, bull, pxc, ext - buf if bull else ext + buf)
                        exits(o, "r_cf_S100", tt, aa, bb, bull, pxc, f1)
                        break
                rows.append(o)
            pd.DataFrame(rows).to_parquet(f)
            print(tf, k, y, len(rows), f"{time.time() - t0:.0f}s", flush=True)
            if time.time() - t_start > 120:
                return
    print("all done")


def pf(s):
    s = s.dropna()
    return round(float(s[s > 0].sum() / -s[s < 0].sum()), 2) if (s < 0).any() else np.nan


def report():
    rows = []
    for tf, k in SETS:
        d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob(f"{tf}_K{k}_20*.parquet"))], ignore_index=True)
        d["yr"] = d.created.dt.year
        d["conf_any"] = d.conf_old | d.conf_day | d.conf_sess
        filt = {"all": d.index == d.index, "OLD zone": d.conf_old, "DAY level": d.conf_day, "SESS level": d.conf_sess,
                "ANY conf": d.conf_any, "OLD+(DAY|SESS)": d.conf_old & (d.conf_day | d.conf_sess)}
        for fn, fm in filt.items():
            for c in [c for c in d.columns if c.startswith("r_")]:
                x = d[fm]
                a, b = x[x.yr <= 2020][c].dropna(), x[x.yr >= 2021][c].dropna()
                rows.append(dict(set=f"{tf} K{k}", filter=fn, exit=c[2:], tr_yr_in=round(len(a) / 5), pf_16_20=pf(a),
                                 R_in=round(a.sum(), 1), tr_yr_out=round(len(b) / 5.75), pf_21_26=pf(b), R_out=round(b.sum(), 1)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "conf_report.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    for s, g in r.groupby("set"):
        print(f"\n=== {s} === (top 15 by 2016-20 PF, >= 15 trades/yr)")
        print(g[g.tr_yr_in >= 15].sort_values("pf_16_20", ascending=False).head(15).drop(columns="set").to_string(index=False))
        print("  baseline all/lim_S100_1R:", g[(g["filter"] == "all") & (g.exit == "lim_S100_1R")][["pf_16_20", "pf_21_26"]].values.tolist())


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
