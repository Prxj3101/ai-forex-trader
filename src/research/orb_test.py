r"""
ORB (Opening Range Breakout) TEST on REAL TICKS + combos with the zone strategy. Rules fixed before running.

RANGES (New York time, from closed M5 bid candles):
  ASIA  : 19:00-02:00 range, trade window 02:00-11:00
  LON30 : 03:00-03:30 range, trade window 03:30-11:00
  NY30  : 08:00-08:30 range, trade window 08:30-12:00
ORB TRADE: first tick that breaks the range in the window (BUY: bid > range high -> fill at that tick's ask,
  SELL: bid < range low -> fill at bid). Only the first break per range per day.
  SL: other side of the range ("full") or range middle ("mid"). TP: 1R, 2R, or session end (16:00 NY).
  Exits tick by tick (stop at the real tick price). Costs $5/lot + 0.2 pip on stop exits.
  Filters (side chosen on 2016-20): range size vs daily ATR (bottom 40% / top 40%), H4 trend (last H4 BOS).
COMBO 1  zone trades (tick labels) only in the direction of the latest range break before the fill, same day.
COMBO 2  BREAK + RETEST: zones CREATED after the break, same direction, inside the window, with the broken range
         level inside the zone +-1 ATR (and a looser version without the level condition).
PASS: 2021-26 net PF >= 1.10, >= 5 of 6 years positive, >= 2 of 3 pairs > 1, >= 150 trades.
Chalana: python src\research\orb_test.py run EURUSD 2016 2018 | python src\research\orb_test.py report
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr, structure_breaks  # noqa: E402

OUT = ROOT / "data/research/orb"
NY = "America/New_York"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
RANGES = {"ASIA": ((19, 0), (2, 0), (11, 0)), "LON30": ((3, 0), (3, 30), (11, 0)), "NY30": ((8, 0), (8, 30), (12, 0))}
EOD = (16, 0)
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 2) if (r < 0).any() else np.nan  # noqa: E731


def ms(ts):
    return int(ts.value // 1_000_000)


def exit_sim(t, a, b, bull, entry, sl, tp_r, end_ms):
    """Tick exit: SL at real tick price, TP exact (tp_r None = no TP), else close at end_ms. Returns (r, why)."""
    risk = abs(entry - sl)
    m = t < end_ms
    t, a, b = t[m], a[m], b[m]
    if len(t) == 0:
        return np.nan, "missing"
    px = b if bull else a
    fav = ((px - entry) if bull else (entry - px)) / risk
    hs = np.nonzero(fav <= -1)[0]; ht = np.nonzero(fav >= tp_r)[0] if tp_r else np.array([], int)
    i_s = hs[0] if len(hs) else len(fav); i_t = ht[0] if len(ht) else len(fav)
    if i_s < len(fav) and i_s <= i_t:
        return float(fav[i_s]), "sl"
    if i_t < len(fav):
        return float(tp_r), "tp"
    return float(fav[-1]), "time"


def run(sym, y0, y1):
    m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["high", "low"])
    d1 = pd.read_parquet(ROOT / f"data/processed/{sym}_D1.parquet"); d1["atr"] = atr(d1)
    d1_ct = pd.DatetimeIndex(d1["close_time"]).values
    h4 = pd.read_parquet(ROOT / f"data/processed/{sym}_H4.parquet")
    br = structure_breaks(h4, 2)
    br_t = pd.DatetimeIndex(br["time"]); br_d = br["direction"].astype(str).to_numpy()
    ny_idx = m5.index.tz_convert(NY)
    rows = []
    days = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D", tz=NY)
    for day in days:
        if day.weekday() >= 5:
            continue
        for name, ((h0, m0), (h1, m1), (h2, m2)) in RANGES.items():
            r_start = (day - pd.Timedelta(days=1) if h0 > h1 else day) + pd.Timedelta(hours=h0, minutes=m0)
            r_end = day + pd.Timedelta(hours=h1, minutes=m1)
            w_end = day + pd.Timedelta(hours=h2, minutes=m2)
            eod = day + pd.Timedelta(hours=EOD[0], minutes=EOD[1])
            sel = (ny_idx >= r_start) & (ny_idx < r_end)
            if sel.sum() < 3:
                continue
            hi, lo = float(m5.high.values[sel].max()), float(m5.low.values[sel].min())
            tk = te.ticks(sym, ms(r_end.tz_convert("UTC")), ms(eod.tz_convert("UTC")))
            if tk is None or len(tk[0]) == 0:
                continue
            t, a, b = tk
            wmask = t < ms(w_end.tz_convert("UTC"))
            up = np.nonzero(wmask & (b > hi))[0]; dn = np.nonzero(wmask & (b < lo))[0]
            iu = up[0] if len(up) else 10**12; idn = dn[0] if len(dn) else 10**12
            if iu == 10**12 and idn == 10**12:
                rows.append(dict(symbol=sym, day=day.date(), range=name, hi=hi, lo=lo, broke=0)); continue
            bull = iu < idn
            i = iu if bull else idn
            entry = float(a[i] if bull else b[i])
            di = np.searchsorted(d1_ct, np.datetime64(r_end.tz_convert("UTC").tz_localize(None), "ns"), side="right") - 1
            datr = float(d1.atr.values[di]) if di >= 0 else np.nan
            k = br_t.searchsorted(pd.Timestamp(t[i], unit="ms", tz="UTC"), side="right") - 1
            trend = (1 if br_d[k] == "bull" else -1) if k >= 0 else 0
            out = dict(symbol=sym, day=day.date(), range=name, hi=hi, lo=lo, broke=1, direction="bull" if bull else "bear",
                       break_ms=int(t[i]), entry=entry, range_atr=(hi - lo) / datr if datr else np.nan,
                       trend_aligned=int(trend == (1 if bull else -1)))
            for slname, sl in (("full", lo if bull else hi), ("mid", (hi + lo) / 2)):
                risk = (entry - sl) if bull else (sl - entry)
                if risk <= 0:
                    continue
                out[f"{slname}_risk"] = risk
                for tpn, tp in (("1R", 1.0), ("2R", 2.0), ("eod", None)):
                    r, why = exit_sim(t[i:], a[i:], b[i:], bull, entry, sl, tp, ms(eod.tz_convert("UTC")))
                    out[f"{slname}_{tpn}"], out[f"{slname}_{tpn}_why"] = r, why
            rows.append(out)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"orb_{sym}_{y0}_{y1}.parquet")
    print(sym, y0, y1, len(d), "ranges", int(d.broke.sum()), "breaks", flush=True)


def load_orb():
    return pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("orb_*.parquet"))], ignore_index=True)


def judge(x, col):
    x = x[x[col].notna()]
    y = pd.to_datetime(x["when"]).dt.year
    yrs = x.groupby(y)[col].sum()
    pairs = [pf(q[col]) for _, q in x.groupby("symbol")]
    ok = len(x) >= 150 and pf(x[col]) >= 1.10 and (yrs > 0).sum() >= 5 and sum(p > 1 for p in pairs) >= 2
    return dict(n=len(x), win=round(100 * float((x[col] > 0).mean()), 1), pf=pf(x[col]),
                pos_years=f"{int((yrs > 0).sum())}/{yrs.size}", pairs=pairs, PASS=ok)


def net_orb(d):
    for sl in ("full", "mid"):
        risk = d[f"{sl}_risk"]
        per_lot = np.where(d.symbol.str.endswith("USD"), risk * 1e5, risk * 1e5 / d.entry)
        cost, slip = 5.0 / per_lot, 0.2 * d.symbol.map(PIP) / risk
        for tp in ("1R", "2R", "eod"):
            c = f"{sl}_{tp}"
            d[c + "_net"] = d[c] - cost - np.where(d[c + "_why"] == "sl", slip, 0)
    return d


def report_orb():
    d = load_orb()
    d = net_orb(d[d.broke == 1].copy())
    d["when"] = pd.to_datetime(d.break_ms, unit="ms")
    rows = []
    for rng in RANGES:
        g = d[d.range == rng]
        learn, test = g[g.when.dt.year <= 2020], g[g.when.dt.year >= 2021]
        for sl in ("full", "mid"):
            for tp in ("1R", "2R", "eod"):
                c = f"{sl}_{tp}_net"
                rows.append(dict(range=rng, sl=sl, tp=tp, filter="(none)", learn_pf=pf(learn[c].dropna()), **judge(test, c)))
                q = learn.range_atr.quantile([0.4, 0.6]).values
                small = pf(learn.loc[learn.range_atr <= q[0], c].dropna()) >= pf(learn.loc[learn.range_atr >= q[1], c].dropna())
                mt = (test.range_atr <= q[0]) if small else (test.range_atr >= q[1])
                rows.append(dict(range=rng, sl=sl, tp=tp, filter=f"range {'small' if small else 'big'}",
                                 learn_pf=pf(learn.loc[(learn.range_atr <= q[0]) if small else (learn.range_atr >= q[1]), c].dropna()),
                                 **judge(test[mt], c)))
                rows.append(dict(range=rng, sl=sl, tp=tp, filter="H4 trend", learn_pf=pf(learn.loc[learn.trend_aligned == 1, c].dropna()),
                                 **judge(test[test.trend_aligned == 1], c)))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "data/research/orb_alone.csv", index=False)
    return t


def combos():
    from src import config
    from src.ml import walkforward as wf
    z = wf.load(config.CORE_SYMBOLS)
    z = z[z.touch_number == 1].copy()
    z["symbol"] = z.symbol.astype(str)
    o = load_orb(); o = o[o.broke == 1].copy()
    o["break_time"] = pd.to_datetime(o.break_ms, unit="ms", utc=True)
    rows = []
    for rng in RANGES:
        ob = o[o.range == rng][["symbol", "day", "direction", "break_time", "hi", "lo"]].rename(columns={"direction": "orb_dir"})
        zz = z.copy()
        zz["day"] = (zz.fill_time.dt.tz_convert(NY)).dt.date
        m = zz.merge(ob, on=["symbol", "day"], how="inner")
        (h2, m2) = RANGES[rng][2]
        w_end = pd.to_datetime(m.day.astype(str)).dt.tz_localize(NY) + pd.Timedelta(hours=h2, minutes=m2)
        in_win = (m.fill_time > m.break_time) & (m.fill_time.dt.tz_convert(NY) <= w_end)
        same = m.direction.astype(str) == m.orb_dir
        m["when"] = m.fill_time
        lvl = np.where(m.orb_dir == "bull", m.hi, m.lo)
        at_lvl = (lvl >= m.bottom - m.atr) & (lvl <= m.top + m.atr)
        created_after = m.created_at > m.break_time
        for tf in ("M15", "H1"):
            ft = m.zone_tf.astype(str) == tf
            for ex in ("r_tp1", "r_tp2", "r_trail"):
                for name, mask in ((f"zones in window, any direction", ft & in_win),
                                   (f"COMBO1 zones WITH {rng} break direction", ft & in_win & same),
                                   (f"zones AGAINST {rng} break", ft & in_win & ~same),
                                   (f"COMBO2 break+retest (zone made after break)", ft & in_win & same & created_after),
                                   (f"COMBO2 break+retest at broken level", ft & in_win & same & created_after & at_lvl)):
                    x = m[mask]
                    a, b = x[x.when.dt.year <= 2020], x[x.when.dt.year >= 2021]
                    rows.append(dict(range=rng, tf=tf, exit=ex, test=name, learn_pf=pf(a[ex]), **judge(b, ex)))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "data/research/orb_combos.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "report":
        print(report_orb().to_string(index=False))
    else:
        print(combos().to_string(index=False))
