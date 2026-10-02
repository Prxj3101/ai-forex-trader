r"""
Zone dataset: har orderflow zone ka "result card" + features.

Ye bot ka "experience" hai. 10 saal ke har zone pe tumhara Setup #1 chalta hai, aur
record hota hai ki kya hua. Yahi table baad mein ML Brain ki training ka data banegi.

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\research\zone_dataset.py --symbol USDJPY --tf M15

Output:
    data/research/{SYMBOL}_{TF}_zones.parquet   (har zone = ek row)
    + terminal pe summary tables

SETUP #1 (Prajwal ka orderflow zone tap) jaisa simulate hota hai:
    Entry  : zone ke BODY top (bullish) / body bottom (bearish) pe LIMIT order
    SL     : zone ki WICK ke paar + buffer (default 0.1 x ATR)
    TP     : 2R (risk ka double)
    Fill   : zone banne ke baad max WAIT_DAYS din tak limit order khula; tab tak fill nahi = "no_fill"
    Hold   : fill ke baad max HOLD_DAYS din; tab tak SL/TP nahi = "timeout" (us waqt ka R)
    Spread : asli spread use hota hai (BUY ask pe bharta hai, SELL position ask pe band hoti hai)
    Ek M5 candle mein SL aur TP dono touch hon -> SL maante hain (conservative)

LOOKAHEAD RULE: saare features sirf zone ke created_at tak ka data dekhte hain.
Result (label) future se aata hai, wo theek hai, kyunki wahi toh seekhna hai.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.smc import atr, find_orderflow_zones, track_zones  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "research"
NY = "America/New_York"
H1_ZONE_LIFE = 500     # H1 zone kitni H1 candles tak "zinda" gina jaye (track_zones ka max_bars bhi 500)

RR = 2.0
SL_BUFFER_ATR = 0.1
WAIT_DAYS = 3
HOLD_DAYS = 3


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load(symbol: str, tf: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{symbol}_{tf}.parquet")


def session_of(ts: pd.Timestamp) -> str:
    tk = ts.tz_convert("Asia/Tokyo").hour
    ld = ts.tz_convert("Europe/London").hour
    ny = ts.tz_convert(NY).hour
    in_l, in_n = 8 <= ld < 17, 8 <= ny < 17
    if in_l and in_n:
        return "London+NY"
    if in_l:
        return "London"
    if in_n:
        return "New York"
    if 9 <= tk < 18:
        return "Asian"
    return "Off-hours"


def last_closed(htf: pd.DataFrame, t: pd.Timestamp) -> int:
    """Higher timeframe ki aakhri candle jo time t tak BAND ho chuki thi (lookahead se bachne ke liye)."""
    return int(htf["close_time"].searchsorted(t, side="right")) - 1


# ----------------------------------------------------------------------
# Features (sirf zone bante waqt tak ka data)
# ----------------------------------------------------------------------
def add_features(z: pd.DataFrame, df: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame, h1_zones: pd.DataFrame) -> pd.DataFrame:
    h1_atr = atr(h1).to_numpy()
    h4_atr = atr(h4).to_numpy()
    closes = df["close"]
    spreads = df["spread"]

    # H1 order flow direction = aakhri H1 BOS kis taraf tha
    h1_bos = h1_zones[["created_at", "direction"]].sort_values("created_at")

    rows = []
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        bos_close = closes.loc[r.bos_time]
        move = (bos_close - r.bottom) if bull else (r.top - bos_close)

        i1 = last_closed(h1, r.created_at)
        i4 = last_closed(h4, r.created_at)
        a1 = h1_atr[i1] if i1 >= 0 else np.nan
        a4 = h4_atr[i4] if i4 >= 0 else np.nan

        entry = r.body_top if bull else r.body_bottom
        pullback = (bos_close - entry) if bull else (entry - bos_close)

        # H1 flow: created_at se pehle ka aakhri H1 zone kis direction ka tha
        k = h1_bos["created_at"].searchsorted(r.created_at, side="right") - 1
        h1_flow = h1_bos["direction"].iloc[k] if k >= 0 else "none"

        # HTF confluence: koi H1 zone same direction ka, jo abhi zinda hai aur price range overlap karta hai.
        # FIX (2 Oct 2026, live replay check): sirf pichle H1_ZONE_LIFE H1 candles mein bane zones.
        # Pehle 2017 ke purane zones bhi "zinda" gine jaate the (track_zones sirf 500 candles dekhta hai,
        # uske baad invalidated_time NaN = hamesha zinda). Live bot ke paas itna purana data hota hi nahi.
        h1_from = h1.index[max(i1 - H1_ZONE_LIFE, 0)] if i1 >= 0 else r.created_at
        cand = h1_zones[
            (h1_zones["direction"] == r.direction)
            & (h1_zones["created_at"] <= r.created_at)
            & (h1_zones["created_at"] >= h1_from)
            & ((h1_zones["invalidated_time"].isna()) | (h1_zones["invalidated_time"] > r.created_at))
            & (h1_zones["bottom"] <= r.top) & (h1_zones["top"] >= r.bottom)
        ]

        rows.append(dict(
            disp_h1=move / a1 if a1 else np.nan,              # "chhota move?" bade TF ke hisaab se
            disp_h4=move / a4 if a4 else np.nan,
            break_margin_atr=((bos_close - r.broken_swing) if bull else (r.broken_swing - bos_close)) / r.atr,
            zone_height_atr=(r.top - r.bottom) / r.atr,
            pullback_atr=pullback / r.atr,                     # entry tak price ko kitna wapas aana padega
            spread_atr=spreads.loc[r.bos_time] / r.atr,
            h1_flow=h1_flow,
            h1_flow_aligned=h1_flow == r.direction,
            h1_confluence=len(cand) > 0,
            session_created=session_of(r.created_at),
            hour_ny=r.created_at.tz_convert(NY).hour,
            weekday=r.created_at.tz_convert(NY).weekday(),
            year=r.created_at.year,
        ))
    return pd.concat([z.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ----------------------------------------------------------------------
# Trade simulation (M5 candles pe, zyada accurate)
# ----------------------------------------------------------------------
def simulate(z: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    lo, hi, cl, sp = (m5[c].to_numpy() for c in ["low", "high", "close", "spread"])
    idx = m5.index
    wait_bars = WAIT_DAYS * 288
    hold_bars = HOLD_DAYS * 288
    out = []

    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        entry = r.body_top if bull else r.body_bottom
        buf = SL_BUFFER_ATR * r.atr
        sl = (r.bottom - buf) if bull else (r.top + buf)
        risk = abs(entry - sl)
        tp = entry + RR * risk if bull else entry - RR * risk
        res = dict(entry=entry, sl=sl, tp=tp, risk=risk, outcome="no_fill", r_multiple=0.0,
                   fill_time=pd.NaT, exit_time=pd.NaT, bars_to_fill=np.nan, session_fill=None)

        s = int(idx.searchsorted(r.created_at))
        e = min(s + wait_bars, len(m5))
        if risk <= 0 or s >= len(m5):
            out.append(res)
            continue

        # Fill: BUY limit tab bharta hai jab ASK (= bid + spread) entry tak aaye
        if bull:
            hit = np.nonzero(lo[s:e] + sp[s:e] <= entry)[0]
        else:
            hit = np.nonzero(hi[s:e] >= entry)[0]
        if hit.size == 0:
            out.append(res)
            continue
        f = s + int(hit[0])
        res.update(fill_time=idx[f], bars_to_fill=int(hit[0]), session_fill=session_of(idx[f]))

        # Exit: fill wali candle se hi SL/TP check (conservative)
        g = min(f + hold_bars, len(m5))
        if bull:     # long position BID pe band hoti hai
            sl_hit = lo[f:g] <= sl
            tp_hit = hi[f:g] >= tp
        else:        # short position ASK pe band hoti hai
            sl_hit = hi[f:g] + sp[f:g] >= sl
            tp_hit = lo[f:g] + sp[f:g] <= tp
        first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else None
        first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else None

        if first_sl is not None and (first_tp is None or first_sl <= first_tp):
            res.update(outcome="loss", r_multiple=-1.0, exit_time=idx[f + first_sl])
        elif first_tp is not None:
            res.update(outcome="win", r_multiple=RR, exit_time=idx[f + first_tp])
        else:
            last = g - 1
            px = cl[last] if bull else cl[last] + sp[last]
            rm = ((px - entry) if bull else (entry - px)) / risk
            res.update(outcome="timeout", r_multiple=round(float(rm), 3), exit_time=idx[last])
        out.append(res)

    return pd.concat([z.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
def stats(d: pd.DataFrame) -> pd.Series:
    t = d[d["outcome"] != "no_fill"]
    wins = (t["outcome"] == "win").sum()
    gross_win = t.loc[t["r_multiple"] > 0, "r_multiple"].sum()
    gross_loss = -t.loc[t["r_multiple"] < 0, "r_multiple"].sum()
    return pd.Series({
        "zones": len(d),
        "trades": len(t),
        "win_%": round(100 * wins / len(t), 1) if len(t) else np.nan,
        "avg_R": round(t["r_multiple"].mean(), 3) if len(t) else np.nan,
        "total_R": round(t["r_multiple"].sum(), 1),
        "PF": round(gross_win / gross_loss, 2) if gross_loss else np.nan,
    })


def summary(ds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = ds.copy()
    d["disp_h1_bucket"] = pd.cut(d["disp_h1"], [0, 0.5, 1, 1.5, 2.5, 100], labels=["<0.5", "0.5-1", "1-1.5", "1.5-2.5", ">2.5"])
    d["break_bucket"] = pd.cut(d["break_margin_atr"], [-1, 0.25, 0.75, 1.5, 100], labels=["<0.25", "0.25-0.75", "0.75-1.5", ">1.5"])
    tables = {"overall": stats(d).to_frame("all").T}
    for col in ["disp_h1_bucket", "break_bucket", "h1_flow_aligned", "h1_confluence", "has_fvg", "session_fill", "direction", "year"]:
        tables[col] = d.groupby(col, observed=True).apply(stats, include_groups=False)
    return tables


def build(symbol: str, tf: str, swing_n: int, min_disp: float, log=None) -> pd.DataFrame:
    """log(agent, message, progress) = dashboard ko live batane ke liye (optional). Na ho toh print."""
    say = log or (lambda agent, msg, pct=None: print(f"[{agent}] {msg}"))
    t0 = time.time()
    df, m5, h1, h4 = load(symbol, tf), load(symbol, "M5"), load(symbol, "H1"), load(symbol, "H4")
    say("data", f"{symbol} {tf}: {len(df):,} candles + M5/H1/H4 loaded", 100)

    z = find_orderflow_zones(df, swing_n=swing_n, min_disp_atr=min_disp)
    z = track_zones(z, df)
    say("zone", f"{len(z):,} orderflow zones found", 30)

    h1_zones = track_zones(find_orderflow_zones(h1, swing_n=swing_n, min_disp_atr=1.5), h1)
    ds = add_features(z, df, h1, h4, h1_zones)
    say("structure", f"Features computed ({len(h1_zones):,} H1 zones for confluence)", 100)

    say("zone", f"Simulating {len(ds):,} trades on M5 with real spread", 60)
    ds = simulate(ds, m5)
    ds["symbol"], ds["tf"], ds["swing_n"], ds["min_disp"] = symbol, tf, swing_n, min_disp
    say("zone", f"Trades simulated in {time.time() - t0:.1f}s", 100)
    return ds


def save(ds: pd.DataFrame, symbol: str, tf: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"{symbol}_{tf}_zones.parquet"
    ds.to_parquet(out_file)
    return out_file


def main() -> None:
    p = argparse.ArgumentParser(description="Orderflow zone dataset + Setup #1 results")
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--tf", default="M15")
    p.add_argument("--swing", type=int, default=1)
    p.add_argument("--disp", type=float, default=2.0)
    a = p.parse_args()

    ds = build(a.symbol, a.tf, a.swing, a.disp)
    out_file = save(ds, a.symbol, a.tf)
    print(f"[5] saved -> {out_file}\n")

    pd.set_option("display.width", 160)
    print("Break-even win rate at 1:2 = 33.3%  (costs included in simulation)\n")
    for name, table in summary(ds).items():
        print(f"== {name} ==")
        print(table.to_string(), "\n")


if __name__ == "__main__":
    main()
