r"""
ENTRY LEVEL INSIDE THE ZONE (real ticks). Same zones as the bot (find_orderflow_zones, swing 1,
displacement per pair from config), first touch only, M15 and H1 zones, 2016-2026.
Entries (BUY example; SELL mirrored):
  wick_edge  = zone top (wick high of the red candle)          -> fills most, smallest SL
  body_edge  = body top (CURRENT bot)
  zone_mid   = middle of the whole zone (wick to wick)
  body_mid   = middle of the candle body
  body_far   = body bottom                                      -> fills least, deepest entry
SL = far wick + 0.1 ATR for all. Limit order live from zone creation until the zone dies (M5 close beyond the
far wick) or 5 days. Fill + exits on ticks. Exits 1:1, 1:2, trail. Costs $5/lot + 0.2 pip on stops.
Rules fixed before: compare on 2016-20 and 2021-26 separately; PASS = 2021-26 net PF >= 1.10, >= 5/6 yrs, >= 2/3 pairs.
Chalana: python src\research\entry_test.py run EURUSD 2016 2018 | python src\research\entry_test.py report
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.backtest import tick_engine as te  # noqa: E402
from src.features.smc import atr, find_orderflow_zones  # noqa: E402

OUT = ROOT / "data/research/entry_test"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
ENTRIES = ["wick_edge", "body_edge", "zone_mid", "body_mid", "body_far"]
LIFE_MS = 5 * 86_400_000
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 2) if (r < 0).any() else np.nan  # noqa: E731


def zones(sym, tf):
    f = OUT / f"zones_{sym}_{tf}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    m = pd.read_parquet(ROOT / f"data/processed/{sym}_{tf}.parquet")
    z = find_orderflow_zones(m, swing_n=config.SWING_N, min_disp_atr=config.disp_for(sym))
    z.to_parquet(f)
    return z


def run(sym, y0, y1):
    m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["close"])
    t5 = m5.index.as_unit("ns").asi8 // 1_000_000; c5 = m5.close.to_numpy()
    rows = []
    for tf in ("M15", "H1"):
        z = zones(sym, tf)
        z = z[(pd.to_datetime(z.created_at).dt.year >= y0) & (pd.to_datetime(z.created_at).dt.year <= y1)]
        for r in z.itertuples(index=False):
            bull = r.direction == "bull"
            c_ms = int(pd.Timestamp(r.created_at).value // 1_000_000)
            i0, i1 = np.searchsorted(t5, c_ms), np.searchsorted(t5, c_ms + LIFE_MS)
            dead = (c5[i0:i1] < r.bottom) if bull else (c5[i0:i1] > r.top)
            end_ms = int(t5[i0 + int(np.argmax(dead))] + 300_000) if dead.any() else c_ms + LIFE_MS
            sl = r.bottom - 0.1 * r.atr if bull else r.top + 0.1 * r.atr
            lv = dict(wick_edge=r.top if bull else r.bottom, body_edge=r.body_top if bull else r.body_bottom,
                      zone_mid=(r.top + r.bottom) / 2, body_mid=(r.body_top + r.body_bottom) / 2,
                      body_far=r.body_bottom if bull else r.body_top)
            tk = te.ticks(sym, c_ms, end_ms + te.HOLD_MS)
            if tk is None:
                continue
            T, A, B = tk
            w = T < end_ms
            for name, entry in lv.items():
                risk = (entry - sl) if bull else (sl - entry)
                base = dict(symbol=sym, tf=tf, entry_type=name, direction=r.direction, created_ms=c_ms,
                            entry=entry, sl=sl, sl_pips=risk / PIP[sym])
                if risk <= 0:
                    continue
                hit = np.nonzero(w & ((A <= entry) if bull else (B >= entry)))[0]
                if not len(hit):
                    rows.append(dict(base, status="no_fill")); continue
                i = int(hit[0]); f = int(T[i])
                j = int(np.searchsorted(T, f + te.HOLD_MS))
                out = dict(base, status="ok", fill_ms=f)
                for p in ("tp1", "tp2", "trail"):
                    x = te.exit_arrays(T[i:j], A[i:j], B[i:j], bull, entry, sl, p)
                    out[f"r_{p}"], out[f"why_{p}"] = x["r"], x["reason"]
                rows.append(out)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"res_{sym}_{y0}_{y1}.parquet")
    print(sym, y0, y1, d.groupby("entry_type").status.apply(lambda s: (s == "ok").sum()).to_dict(), flush=True)


def report():
    d = pd.concat([pd.read_parquet(f) for f in OUT.glob("res_*.parquet")], ignore_index=True)
    total = d.groupby(["tf", "entry_type"]).size()
    d = d[d.status == "ok"].reset_index(drop=True)
    d["year"] = pd.to_datetime(d.fill_ms, unit="ms").dt.year
    risk = d.sl_pips * d.symbol.map(PIP)
    per_lot = np.where(d.symbol.str.endswith("USD"), risk * 1e5, risk * 1e5 / d.entry)
    cost, slip = pd.Series(5.0 / per_lot, index=d.index), 0.2 * d.symbol.map(PIP) / risk
    rows = []
    for (tf, et), g in d.groupby(["tf", "entry_type"]):
        idx = g.index
        for p in ("tp1", "tp2", "trail"):
            n = g[f"r_{p}"] - cost[idx] - np.where(g[f"why_{p}"].isin(["sl", "trail", "be"]), slip[idx], 0)
            a, b = n[g.year <= 2020], n[g.year >= 2021]
            yrs = n[g.year >= 2021].groupby(g.year[g.year >= 2021]).sum()
            pairs = [pf(n[g.symbol == s][g.year[g.symbol == s] >= 2021]) for s in PIP]
            rows.append(dict(tf=tf, entry=et, exit=p, fill_pct=round(100 * len(g) / total[(tf, et)], 1),
                             trades=len(g), sl_pips=round(g.sl_pips.median(), 1),
                             win=round(100 * float((g[f"r_{p}"] > 0).mean()), 1), gross_pf=pf(g[f"r_{p}"]),
                             pf_2016_20=pf(a), pf_2021_26=pf(b), pos_years=f"{int((yrs > 0).sum())}/{yrs.size}", pairs=pairs,
                             PASS=bool(pf(b) >= 1.10 and (yrs > 0).sum() >= 5 and sum(x > 1 for x in pairs) >= 2)))
    t = pd.DataFrame(rows)
    order = {e: i for i, e in enumerate(ENTRIES)}
    t = t.sort_values(["tf", "exit", "entry"], key=lambda s: s.map(order) if s.name == "entry" else s)
    t.to_csv(ROOT / "data/research/entry_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        report()
