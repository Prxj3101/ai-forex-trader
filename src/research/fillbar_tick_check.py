r"""
FILL-CANDLE CHECK with real ticks (research only; live bot / models untouched).

Problem (ChatGPT audit, point 1): old backtest used the WHOLE fill M5 candle's high/low,
but the limit order fills somewhere inside that candle -> price moves BEFORE the fill were
counted as profit (TP / trailing best).

This script: for every bot trade (M15 touch1 + ML top30 + VA + activity filters, 2019-2026)
  1. pulls the real Dukascopy ticks of the fill candle (JForex CSV, EET time -> UTC)
  2. finds the actual fill tick (BUY: ask <= entry, SELL: bid >= entry)
  3. replays SL / TP / best price tick-by-tick from the fill to the candle end
  4. continues on M5 candles after that (same rules as before: stop first, trail from next candle)
Steps:  python src\research\fillbar_tick_check.py extract EURUSD 2019     (grep ticks of fill candles)
        python src\research\fillbar_tick_check.py replay                  (all pairs, uses extracted ticks)
"""
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.risk import portfolio_sim as ps  # noqa: E402

TICKS = ROOT / "data" / "ticks" / "jforex"
OUT = ROOT / "data" / "research" / "fillbar_ticks"
EET = "Europe/Helsinki"


def bot_trades() -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "data/research/ml_scores.parquet")
    for c in ("symbol", "zone_tf", "direction"):
        s[c] = s[c].astype(str)
    s = s[(s.zone_tf == "M15") & (s.touch_number == 1) & s["keep_0.3"]
          & config.va_ok(s.r24_va_pos) & config.act_ok(s.act_15m)]
    return s.reset_index(drop=True)


def year_file(sym, year):
    f = TICKS / f"{sym}_Ticks_{year}.01.01_{year + 1}.01.01.csv"
    if not f.exists():
        cands = sorted(TICKS.glob(f"{sym}_Ticks_{year}.01.01_*.csv"), key=lambda p: p.stat().st_size)
        f = cands[-1]
    return f


def extract(sym, year):
    s = bot_trades()
    s = s[(s.symbol == sym) & (s.fill_time.dt.year == year)]
    pats = set()
    for t in s.fill_time:
        for m in range(5):
            loc = (t + pd.Timedelta(minutes=m)).tz_convert(EET)
            pats.add(loc.strftime("%Y.%m.%d %H:%M:"))
    pf = OUT / f"{sym}_{year}.pat"
    pf.write_text("\n".join(sorted(pats)) + "\n")
    out = OUT / f"{sym}_{year}.csv"
    with open(out, "w") as fh:
        subprocess.run(["grep", "-F", "-f", str(pf), str(year_file(sym, year))], stdout=fh, check=False)
    print(sym, year, len(s), "trades", sum(1 for _ in open(out)), "ticks")


def load_ticks(sym):
    parts = []
    for f in sorted(OUT.glob(f"{sym}_*.csv")):
        d = pd.read_csv(f, header=None, names=["t", "ask", "bid", "av", "bv"], usecols=[0, 1, 2])
        parts.append(d)
    d = pd.concat(parts)
    t = pd.to_datetime(d["t"], format="%Y.%m.%d %H:%M:%S.%f")
    d.index = t.dt.tz_localize(EET, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    d = d[d.index.notna()].sort_index()
    return d[["ask", "bid"]]


def replay(save=True):
    s = bot_trades()
    rows = []
    for sym, g in s.groupby("symbol"):
        tk = load_ticks(sym)
        m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["high", "low", "close", "spread"])
        hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
        idx = m5.index
        for r in g.itertuples(index=False):
            bull = r.direction == "bull"
            risk = abs(r.entry - r.sl)
            f0 = int(idx.searchsorted(r.fill_time))
            g1 = min(f0 + ps.HOLD_BARS, len(m5))
            w = tk.loc[r.fill_time: r.fill_time + pd.Timedelta(minutes=5) - pd.Timedelta(microseconds=1)]
            base = dict(symbol=sym, direction=r.direction, fill_time=r.fill_time, entry=r.entry, sl=r.sl,
                        exit=r.exit_ml, n_ticks=len(w))
            if len(w) == 0:
                rows.append(dict(base, status="no_ticks")); continue
            ask, bid = w["ask"].to_numpy(), w["bid"].to_numpy()
            hit = np.nonzero(ask <= r.entry if bull else bid >= r.entry)[0]
            if len(hit) == 0:
                rows.append(dict(base, status="no_fill")); continue
            j = hit[0]
            px = bid[j:] if bull else ask[j:]                       # exit side
            fav = (px - r.entry) / risk if bull else (r.entry - px) / risk
            # pre-fill best (what the old code could see)
            out = dict(base, status="ok", fill_tick=w.index[j], pre_fill_best=float(
                ((bid[:j] - r.entry) / risk).max() if bull and j else ((r.entry - ask[:j]) / risk).max() if j else np.nan))
            for ex in ("tp1", "trail"):
                p = ps.EXIT_POLICY[ex]
                res = None
                best = 0.0
                a = np.nonzero(fav <= -1.0)[0]; b = np.nonzero(fav >= p.tp_r)[0] if p.tp_r is not None else np.array([], int)
                ia = a[0] if len(a) else len(fav); ib = b[0] if len(b) else len(fav)
                if ia < len(fav) and ia <= ib:
                    res = (-1.0, "sl", 0)
                elif ib < len(fav):
                    res = (p.tp_r, "tp", 0)
                best = max(0.0, float(fav.max()))
                if res is None:
                    sl_r = -1.0
                    if p.trail_r is not None and best >= p.trail_start_r:
                        sl_r = max(sl_r, best - p.trail_r)
                    for i in range(f0 + 1, g1):
                        if bull:
                            fv = (hi[i] - r.entry) / risk; ad = (lo[i] - r.entry) / risk
                        else:
                            fv = (r.entry - (lo[i] + sp[i])) / risk; ad = (r.entry - (hi[i] + sp[i])) / risk
                        if ad <= sl_r:
                            res = (sl_r, "sl" if sl_r < -0.01 else ("be" if abs(sl_r) <= 0.01 else "trail"), i - f0); break
                        if p.tp_r is not None and fv >= p.tp_r:
                            res = (p.tp_r, "tp", i - f0); break
                        best = max(best, fv)
                        if p.trail_r is not None and best >= p.trail_start_r:
                            sl_r = max(sl_r, best - p.trail_r)
                    if res is None:
                        x = cl[g1 - 1] if bull else cl[g1 - 1] + sp[g1 - 1]
                        res = (((x - r.entry) if bull else (r.entry - x)) / risk, "timeout", g1 - 1 - f0)
                out[f"{ex}_r"], out[f"{ex}_why"], out[f"{ex}_k"] = res
            rows.append(out)
    d = pd.DataFrame(rows)
    if save:
        d.to_parquet(ROOT / "data/research/fillbar_tick_check.parquet")
    return d


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], int(sys.argv[3]))
    else:
        d = replay()
        print(d.status.value_counts())
