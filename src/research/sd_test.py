r"""
SUPPLY & DEMAND (YouTube video strategy) on REAL TICKS. Rules written BEFORE testing (he confirmed):

ZONE (M15):
  - displacement = at least 3 CONSECUTIVE big candles of one colour; big = body >= 1.5 x average body of previous 20
  - at least one fair value gap inside the move (3-candle gap)
  - break of structure: a move candle closes beyond the last confirmed swing (swing n=2) before the move
  - zone = last opposite candle (within 3 candles before the move), wick to wick; created at the close of the 3rd+ big candle
  - trend flag: last two confirmed swing lows rising (BUY) / swing highs falling (SELL)  [reported with and without]
LIFE: zone dead on any M15 close beyond the far wick; fresh only (first touch); max 5 days
ENTRIES (each tested separately):
  normal       : touch candle or one of the next 3 closes back OUTSIDE the zone -> market entry next tick
  conservative : engulfing candle (within 4 candles after touch) -> market entry next tick
  aggressive   : limit at middle of zone-candle body; skipped if the candle before the touch was a big candle
                 diving into the zone (falling knife)
SL = far wick + 0.1 ATR. TP = recent high/low (top of the leg before the pullback); skip if TP < 1R.
     Also fixed 1:2 reported. Session: entry between 02:00 and 16:00 New York (London + NY).
Costs: $5/lot + 0.2 pip on stop exits.   Pass mark: net PF > 1.10 and > 1 in both 2016-20 and 2021-26.
Chalana:  python src\research\sd_test.py run EURUSD 2019     |   python src\research\sd_test.py report
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr, find_swings  # noqa: E402

import os
TF = os.environ.get("SD_TF", "M15")
TF_MIN = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}[TF]
OUT = ROOT / ("data/research/sd_test" + ("" if TF == "M15" else f"_{TF}"))
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
COMM, SLIP = 5.0, 0.2
LIFE_BARS = 5 * 24 * 60 // TF_MIN
NY = "America/New_York"


def zones(sym):
    f = OUT / f"zones_{sym}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    m = pd.read_parquet(ROOT / f"data/processed/{sym}_{TF}.parquet")
    o, h, l, c = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(m).to_numpy()
    body = np.abs(c - o)
    avgb = pd.Series(body).rolling(20).mean().shift(1).to_numpy()
    big = body >= 1.5 * avgb
    green, red = c > o, c < o
    is_sh, is_sl = find_swings(m, 2)
    sh, sl = [], []                       # confirmed swings (index, value)
    rows, run_b, run_s = [], 0, 0
    used_b = used_s = -1
    for i in range(len(m)):
        k = i - 2
        if k >= 0:
            if is_sh[k]: sh.append((k, h[k]))
            if is_sl[k]: sl.append((k, l[k]))
        run_b = run_b + 1 if (big[i] and green[i]) else 0
        run_s = run_s + 1 if (big[i] and red[i]) else 0
        for bull, run in ((True, run_b), (False, run_s)):
            if run < 3:
                continue
            s0 = i - run + 1
            if (used_b if bull else used_s) == s0:
                continue
            piv = [v for (j, v) in (sh if bull else sl) if j < s0]
            if not piv:
                continue
            bos = (c[s0:i + 1].max() > piv[-1]) if bull else (c[s0:i + 1].min() < piv[-1])
            fvg = any((l[j + 1] > h[j - 1]) if bull else (h[j + 1] < l[j - 1]) for j in range(s0, i))
            zc = next((j for j in range(s0 - 1, max(s0 - 4, -1), -1) if (red[j] if bull else green[j])), None)
            if not (bos and fvg and zc is not None and a[zc] > 0):
                continue
            other = [v for (j, v) in (sl if bull else sh) if j < s0]
            trend = len(other) >= 2 and ((other[-1] > other[-2]) if bull else (other[-1] < other[-2]))
            if bull: used_b = s0
            else: used_s = s0
            rows.append(dict(direction="bull" if bull else "bear", zc=zc, created=i, top=h[zc], bottom=l[zc],
                             mid=(o[zc] + c[zc]) / 2, atr=a[zc], trend=trend, run=run))
    z = pd.DataFrame(rows)
    z["created_time"] = m.index[z.created.to_numpy()]
    OUT.mkdir(parents=True, exist_ok=True)
    z.to_parquet(f)
    return z


def run(sym, year):
    m = pd.read_parquet(ROOT / f"data/processed/{sym}_{TF}.parquet")
    o, h, l, c = (m[k].to_numpy() for k in ("open", "high", "low", "close"))
    body = np.abs(c - o)
    avgb = pd.Series(body).rolling(20).mean().shift(1).to_numpy()
    idx = m.index
    end_ms = (idx + pd.Timedelta(minutes=TF_MIN)).as_unit("ns").asi8 // 1_000_000
    z = zones(sym)
    z = z[z.created_time.dt.year == year]
    rows = []
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        i0, i1 = r.created + 1, min(r.created + 1 + LIFE_BARS, len(m))
        if i0 >= len(m):
            continue
        near, far = (r.top, r.bottom) if bull else (r.bottom, r.top)
        sl = far - 0.1 * r.atr if bull else far + 0.1 * r.atr
        touch = (l[i0:i1] <= near) if bull else (h[i0:i1] >= near)
        dead = (c[i0:i1] < far) if bull else (c[i0:i1] > far)
        t = i0 + int(np.argmax(touch)) if touch.any() else None
        d_i = i0 + int(np.argmax(dead)) if dead.any() else 10**9
        base = dict(symbol=sym, direction=r.direction, created_time=r.created_time, trend=r.trend, sl=sl)
        if t is None:
            rows.append(dict(base, model="none", status="no_touch")); continue
        leg = h[r.created:t].max() if bull else l[r.created:t].min()      # recent high/low before pullback
        knife = t - 1 >= 0 and body[t - 1] >= 2 * avgb[t - 1] and ((c[t - 1] < o[t - 1]) if bull else (c[t - 1] > o[t - 1]))
        entries = {}
        # normal: touch candle or next 3 close back outside (before any close beyond far wick)
        for k in range(t, min(t + 4, len(m))):
            if k >= d_i: break
            if (c[k] > near) if bull else (c[k] < near):
                entries["normal"] = ("mkt", end_ms[k]); break
        # conservative: engulfing within 4 candles after touch
        for k in range(t + 1, min(t + 5, len(m))):
            if k >= d_i: break
            p = k - 1
            eng = ((c[k] > o[k]) and c[k] >= max(o[p], c[p]) and o[k] <= min(o[p], c[p]) and c[p] < o[p]) if bull else \
                  ((c[k] < o[k]) and c[k] <= min(o[p], c[p]) and o[k] >= max(o[p], c[p]) and c[p] > o[p])
            if eng:
                entries["conservative"] = ("mkt", end_ms[k]); break
        # aggressive: limit at body middle, from zone creation until zone death / life end
        if not knife:
            s_ms = end_ms[r.created]; e_ms = end_ms[min(d_i, i1 - 1)]
            f = te.find_fill(sym, bull, r.mid, int(s_ms), int(e_ms))
            if isinstance(f, int):
                entries["aggressive"] = ("lim", f)
        for model, (kind, t_ms) in entries.items():
            out = dict(base, model=model, knife=bool(knife))
            if kind == "mkt":
                tk = te.ticks(sym, int(t_ms), int(t_ms) + 600_000)
                if tk is None or len(tk[0]) == 0:
                    rows.append(dict(out, status="missing")); continue
                e_ms, entry = int(tk[0][0]), float(tk[1][0] if bull else tk[2][0])
            else:
                e_ms, entry = int(t_ms), r.mid
            hr = pd.Timestamp(e_ms, unit="ms", tz="UTC").tz_convert(NY).hour
            risk = (entry - sl) if bull else (sl - entry)
            if risk <= 0:
                rows.append(dict(out, status="bad_risk")); continue
            rr = ((leg - entry) if bull else (entry - leg)) / risk
            out.update(entry_ms=e_ms, entry=entry, sl_pips=risk / PIP[sym], rr=rr, ny_hour=hr,
                       session_ok=2 <= hr < 16, status="ok")
            if rr >= 1:
                x = te.run_exit(sym, bull, entry, sl, e_ms, "tp", tp_r=float(rr))
                out["r_leg"], out["why_leg"] = x["r"], x["reason"]
            x = te.run_exit(sym, bull, entry, sl, e_ms, "tp2")
            out["r_tp2"], out["why_tp2"] = x["r"], x["reason"]
            rows.append(out)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / f"res_{sym}_{year}.parquet")
    print(sym, year, len(z), "zones", d[d.status == "ok"].model.value_counts().to_dict(), flush=True)


def report():
    d = pd.concat([pd.read_parquet(f) for f in OUT.glob("res_*.parquet")], ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["year"] = pd.to_datetime(d.entry_ms, unit="ms").dt.year
    risk = d.sl_pips * d.symbol.map(PIP)
    per_lot = np.where(d.symbol.str.endswith("USD"), risk * 1e5, risk * 1e5 / d.entry)
    cost, slip = COMM / per_lot, SLIP * d.symbol.map(PIP) / risk
    for p in ("leg", "tp2"):
        d[f"n_{p}"] = d[f"r_{p}"] - cost - np.where(d[f"why_{p}"].isin(["sl", "trail", "be"]), slip, 0)
    pf = lambda r: round(r[r > 0].sum() / -r[r < 0].sum(), 2) if (r < 0).any() and len(r) else np.nan  # noqa: E731
    rows = []
    for model in ("normal", "conservative", "aggressive"):
        for tr in (False, True):
            g = d[(d.model == model) & d.session_ok & (d.trend if tr else True)]
            for p in ("leg", "tp2"):
                x = g[g[f"r_{p}"].notna()]
                a, b = x[x.year <= 2020], x[x.year >= 2021]
                rows.append(dict(model=model, trend_filter=tr, target="recent high/low" if p == "leg" else "fixed 1:2",
                                 trades=len(x), win_pct=round(100 * (x[f"r_{p}"] > 0).mean(), 1),
                                 avg_rr=round(x.rr.median(), 2) if p == "leg" else 2.0, sl_pips=round(x.sl_pips.median(), 1),
                                 gross_pf=pf(x[f"r_{p}"]), net_pf=pf(x[f"n_{p}"]), net_R=round(x[f"n_{p}"].sum()),
                                 pf_2016_20=pf(a[f"n_{p}"]), pf_2021_26=pf(b[f"n_{p}"]),
                                 by_pair={s: pf(q[f"n_{p}"]) for s, q in x.groupby("symbol")}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "report.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    {"zones": lambda: [print(s, len(zones(s))) for s in PIP], "run": lambda: run(sys.argv[2], int(sys.argv[3])),
     "report": report}[sys.argv[1]]()
