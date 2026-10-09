r"""
ORB COMPRESSION CONFIRMATION (9 Oct 2026). Rules fixed BEFORE running.
(1) NEW PAIRS AUDUSD + USDCAD (never used in the ORB work) on M5 bars (no ticks there). M5 approximation:
    BUY stop fills at range high + bar spread, SELL at range low; in the fill bar only the SL is checked;
    later bars SL-first if SL and TP in the same bar; EOD = 16:00 NY bar close. The same M5 code is also run on
    EUR/GBP/JPY so we can see how far M5 is from the tick result (calibration).
(2) NEIGHBOUR SETTINGS: NR3..NR7, TP 1R/1.5R/2R/2.5R/3R/EOD, tight cutoff 20-40%.
Same costs ($5/lot + 0.2 pip on stop exits). Same compression definitions as orb_nr_test.py.
CONFIRM if, on the NEW pairs 2016-26, the 4 passing setups have PF > 1 AND compression beats no-filter.
Chalana: python src\research\orb_confirm.py run AUDUSD | python src\research\orb_confirm.py report
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/research/orb_confirm"
NY = "America/New_York"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "AUDUSD": 1e-4, "USDCAD": 1e-4, "NZDUSD": 1e-4,
       "EURJPY": 1e-2, "GBPJPY": 1e-2, "AUDJPY": 1e-2, "EURGBP": 1e-4}
RANGES = {"ASIA": ((19, 0), (2, 0), (11, 0)), "LON30": ((3, 0), (3, 30), (11, 0)), "LON60": ((3, 0), (4, 0), (11, 0)),
          "NY30": ((8, 0), (8, 30), (12, 0)), "NY60": ((8, 0), (9, 0), (12, 0))}
TPS = {"1R": 1.0, "1.5R": 1.5, "2R": 2.0, "2.5R": 2.5, "3R": 3.0, "EOD": None}
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan  # noqa: E731


def trade(H, L, C, S, k0, bull, entry, sl, tp):
    """Bars from fill bar k0 to the end (EOD). Returns (r, why)."""
    risk = abs(entry - sl)
    for k in range(k0, len(H)):
        if bull:
            hit_sl = L[k] <= sl; hit_tp = tp is not None and k > k0 and H[k] >= tp
        else:
            hit_sl = H[k] + S[k] >= sl; hit_tp = tp is not None and k > k0 and L[k] + S[k] <= tp
        if hit_sl:
            return -1.0, "sl"
        if hit_tp:
            return abs(tp - entry) / risk, "tp"
    last = C[-1] if bull else C[-1] + S[-1]
    return ((last - entry) if bull else (entry - last)) / risk, "time"


def run(sym):
    m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["high", "low", "close", "spread"])
    m5["spread"] = m5.spread.clip(lower=0)
    d1 = pd.read_parquet(ROOT / f"data/processed/{sym}_D1.parquet")
    rng = (d1.high - d1.low).to_numpy(); ct = pd.DatetimeIndex(d1.close_time).values
    dh, dl = d1.high.to_numpy(), d1.low.to_numpy()
    ny = m5.index.tz_convert(NY)
    tv = m5.index.values
    H, L, C, S = (m5[c].to_numpy() for c in ("high", "low", "close", "spread"))
    rows = []
    for day in pd.date_range("2016-01-04", "2026-09-29", freq="D", tz=NY):
        if day.weekday() >= 5:
            continue
        for name, ((h0, m0), (h1, m1), (h2, m2)) in RANGES.items():
            rs = (day - pd.Timedelta(days=1) if h0 > h1 else day) + pd.Timedelta(hours=h0, minutes=m0)
            re = day + pd.Timedelta(hours=h1, minutes=m1); we = day + pd.Timedelta(hours=h2, minutes=m2)
            eod = day + pd.Timedelta(hours=16)
            a, b, c, e = np.searchsorted(tv, [np.datetime64(x.tz_convert("UTC").tz_localize(None), "ns") for x in (rs, re, we, eod)])
            if b - a < 5 or e - b < 10:
                continue
            hi, lo = H[a:b].max(), L[a:b].min(); w = hi - lo
            di = np.searchsorted(ct, np.datetime64(re.tz_convert("UTC").tz_localize(None), "ns"), side="right") - 1
            o = dict(symbol=sym, day=day.date(), range=name, w=w, broke=0)
            if di >= 7:
                for n in range(3, 8):
                    o[f"nr{n}"] = int(rng[di] <= rng[di - n + 1:di + 1].min())
                o["inside"] = int(dh[di] <= dh[di - 1] and dl[di] >= dl[di - 1])
            up = np.nonzero(H[b:c] > hi)[0]; dn = np.nonzero(L[b:c] < lo)[0]
            iu = up[0] if len(up) else 10**9; idn = dn[0] if len(dn) else 10**9
            if iu == idn == 10**9 or iu == idn:
                rows.append(o); continue
            bull = iu < idn; k0 = b + (iu if bull else idn)
            entry = hi + S[k0] if bull else lo
            sl = lo if bull else hi + S[k0]
            risk = abs(entry - sl)
            o.update(broke=1, direction="bull" if bull else "bear", entry=entry, risk=risk)
            for tn, k in TPS.items():
                tp = None if k is None else (entry + k * risk if bull else entry - k * risk)
                r, why = trade(H[:e], L[:e], C[:e], S[:e], k0, bull, entry, sl, tp)
                o[f"r_{tn}"], o[f"why_{tn}"] = r, why
            rows.append(o)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"m5_{sym}.parquet")
    print(sym, len(d), int(d.broke.sum()), flush=True)


def load():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("m5_*.parquet"))], ignore_index=True)
    d = d.sort_values(["symbol", "range", "day"]).reset_index(drop=True)
    d["w_ratio"] = d.w / d.groupby(["symbol", "range"]).w.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    pip = d.symbol.map(PIP)
    d["day_ts"] = pd.to_datetime(d.day)
    usd = pd.Series(np.nan, index=d.index)
    usd[d.symbol.str.endswith("USD")] = 1.0                                   # quote USD
    m = d.symbol.str.startswith("USD"); usd[m] = 1.0 / d.entry[m]             # USD base
    for quote, ref, inv in (("JPY", "USDJPY", True), ("GBP", "GBPUSD", False)):
        m = usd.isna() & d.symbol.str.endswith(quote)
        if m.any():                                                            # crosses: convert quote ccy -> USD
            r = pd.read_parquet(ROOT / f"data/processed/{ref}_D1.parquet", columns=["close", "trade_date"])
            rate = r.set_index(pd.to_datetime(r.trade_date)).close.sort_index()
            px = rate.reindex(d.loc[m, "day_ts"], method="ffill").to_numpy()
            usd[m] = (1.0 / px) if inv else px
    per_lot = d.risk.to_numpy() * 1e5 * usd.to_numpy()
    for tn in TPS:
        d[f"n_{tn}"] = d[f"r_{tn}"] - 5.0 / per_lot - np.where(d[f"why_{tn}"] == "sl", 0.2 * pip / d.risk, 0)
    d["year"] = pd.to_datetime(d.day).dt.year
    return d


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])


def session_table(syms=None, tps=("2R", "EOD"), years=(2016, 2026)):
    """PF per pair x range: plain ORB and with compression (NR4 or TIGHT; tight cutoff = 30th pct of that
    pair+range in 2016-20). M5 approximation (~0.03-0.06 PF optimistic vs ticks)."""
    d = load(); d = d[(d.broke == 1) & d.year.between(*years)]
    if syms:
        d = d[d.symbol.isin(syms)]
    q = {k: g[g.year <= 2020].w_ratio.quantile(0.3) for k, g in d.groupby(["symbol", "range"])}
    d["comp"] = (d.nr4 == 1) | (d.w_ratio <= d.set_index(["symbol", "range"]).index.map(q).to_numpy())
    rows = []
    for (s, rg), g in d.groupby(["symbol", "range"]):
        row = dict(pair=s, range=rg)
        for tp in tps:
            c = f"n_{tp}"
            row[f"plain_{tp}"] = pf(g[c]); row[f"comp_{tp}"] = pf(g[g.comp][c])
        row["comp_n_yr"] = round(g.comp.sum() / (years[1] - years[0] + 0.75))
        rows.append(row)
    return pd.DataFrame(rows)
