r"""
GOLD FIB 7 - V3 (14 Oct 2026). Fixes from his round-2 answers (60 setups, 2016-20). Rules fixed BEFORE running.

WHY  His YES picks: PF 1.39 (SL 1.0, 1R) vs NO 0.20. His NO reasons: (1) bot traded a 2nd tap - he trades ONLY the
     first tap. Checked g01/g03/g08/g22: the chart (BID) wick reached 0.705 but the BUY limit needs ASK <= 0.705 and
     missed by the spread ($0.00-0.22), so the bot filled on a later tap. (2) wrong start/end points: his anchors are
     the extreme of the WHOLE move (bot used a small zigzag swing inside it, e.g. g07, g43, g47).
ZONES  v2 zones (ATR zigzag, leg >= 3 ATR, fresh check, H4 EMA50 trend) with zigzag reversal K = 1.5 (v2) | 2.5 | 3.5
       (bigger K = only bigger reversals count as a swing -> start/end at the extreme of the whole move).
FILL   OLD  v2 way: first tick with ask <= 0.705 (BUY) / bid >= 0.705 (SELL) within 10 days (2nd taps allowed).
       FT1  first tap only: first chart touch = first tick with bid <= 0.705 (BUY) / bid >= 0.705 (SELL).
            The limit must fill within 15 min of that touch, else the zone is dead.
       FT2  first tap, limit placed a spread above (BUY): fill at the first chart touch at that tick's ask
            (SELL: same as touch, fill at bid). SL level unchanged, 1R measured from the real fill.
SL/TP  SL B 0.886 | C 1.0 ; TP 1R | 1.5R. Costs $7/lot RT + $0.05 slip on stops, Dukascopy spread. 3-day hold.
TEST   ONLY 2016-2020 first. PASS (fixed): PF >= 1.10 on 2016-20, >= 30 trades/yr, neighbours (other SL/TP) not far off.
       Then FREEZE the best one and run 2021-26 ONCE (`final K MODE`): PF >= 1.10 needed there too.
Chalana:  python src\research\gold_fib_v3.py run          (2016-2020, resumable)
          python src\research\gold_fib_v3.py report
          python src\research\gold_fib_v3.py final 2.5 FT2   (2021-2026, only after freezing)
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

SYM = "XAUUSD"
OUT = ROOT / "data/research/gold_fib_v3"
KS = [1.5, 2.5, 3.5]
MODES = ["OLD", "FT1", "FT2"]
SLS = {"B": 0.886, "C": 1.0}
TPS = {"1R": 1.0, "1.5R": 1.5}
FT1_WIN = 15 * 60_000
DAY = 86_400_000
_zz = gf.zigzag


def zones_k(k, tf, b, sess, h4):
    gf.zigzag = lambda bb, kk=k: _zz(bb, kk)
    try:
        return gf.zones_v2(tf, b, sess, h4)
    finally:
        gf.zigzag = _zz


def exits(bull, fill_ms, fill_px, H, L, out, mode):
    for sk, f in SLS.items():
        sl = H - f * (H - L) if bull else L + f * (H - L)
        risk = (fill_px - sl) if bull else (sl - fill_px)
        for tk, tr in TPS.items():
            if risk <= 0:
                out[f"r_{mode}_{sk}_{tk}"] = np.nan; continue
            res = te.run_exit(SYM, bull, fill_px, sl, fill_ms, "tp", tr)
            out[f"r_{mode}_{sk}_{tk}"] = res["r"] - gf.COMM / (risk * gf.OZ) - \
                (gf.SLIP / risk if res["reason"] in ("sl", "trail", "be") else 0.0)


def trade(z):
    bull, e = bool(z.bull), float(z.entry)
    c_ms = int(z.created.value // 10**6)
    tk = te.ticks(SYM, c_ms, c_ms + gf.VALID_DAYS * DAY)
    if tk is None:
        return dict(status="missing")
    t, a, b = tk
    out = dict(status="ok")
    # OLD
    hit = np.nonzero(a <= e)[0] if bull else np.nonzero(b >= e)[0]
    out["fill_OLD"] = int(t[hit[0]]) if len(hit) else None
    if len(hit):
        exits(bull, int(t[hit[0]]), e, z.H, z.L, out, "OLD")
    # first chart touch (bid bars)
    tch = np.nonzero(b <= e)[0] if bull else np.nonzero(b >= e)[0]
    if not len(tch):
        return out
    i0 = tch[0]
    out["touch_ms"] = int(t[i0])
    # FT1: limit at 0.705 must fill within 15 min of the first touch
    j = np.nonzero(((a <= e) if bull else (b >= e)) & (t >= t[i0]) & (t <= t[i0] + FT1_WIN))[0]
    if len(j):
        out["fill_FT1"] = int(t[j[0]])
        exits(bull, int(t[j[0]]), e, z.H, z.L, out, "FT1")
    # FT2: fill at the first touch at the real price (BUY at ask, SELL at bid)
    px = float(a[i0]) if bull else float(b[i0])
    out["fill_FT2"] = int(t[i0])
    out["px_FT2"] = px
    exits(bull, int(t[i0]), px, z.H, z.L, out, "FT2")
    return out


def run(years=range(2016, 2021), tag=""):
    m5 = pd.read_parquet(ROOT / f"data/processed/{SYM}_M5.parquet", columns=["high", "low", "close_time"])
    sess = gf.session_table(m5)
    h4 = gf.bars("H4")
    OUT.mkdir(parents=True, exist_ok=True)
    for tf in ["M15", "H1"]:
        b = gf.bars(tf)
        for k in KS:
            f = OUT / f"{tag}K{k}_{tf}.parquet"
            if f.exists():
                print(f.name, "already done"); continue
            allz = zones_k(k, tf, b, sess, h4)
            zz = allz[allz.created.dt.year.isin(list(years))].reset_index(drop=True)
            t0 = time.time()
            rows = [{**z.to_dict(), **trade(z)} for _, z in zz.iterrows()]
            pd.DataFrame(rows).to_parquet(f)
            print(f"{f.name}: zones {len(zz)}, {time.time() - t0:.0f}s", flush=True)


def report(tag=""):
    rows = []
    for f in sorted(OUT.glob(f"{tag}K*_*.parquet")):
        if not tag and f.name.startswith("FINAL"):
            continue
        d = pd.read_parquet(f)
        k, tf = f.stem.replace(tag, "").split("_")
        yrs = d.created.dt.year.nunique()
        for m in MODES:
            for sk in SLS:
                for tk in TPS:
                    c = f"r_{m}_{sk}_{tk}"
                    if c not in d:
                        continue
                    r = d[c].dropna()
                    rows.append(dict(K=k[1:], tf=tf, mode=m, sl=sk, tp=tk, trades_yr=round(len(r) / yrs),
                                     win=round(float((r > 0).mean()) * 100) if len(r) else 0, pf=gf.pf(r), R=round(r.sum(), 1)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run()
    elif cmd == "report":
        report()
    elif cmd == "final":
        KS[:] = [float(sys.argv[2])]
        run(range(2021, 2027), tag="FINAL_")
        report(tag="FINAL_")
