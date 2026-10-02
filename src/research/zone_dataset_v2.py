r"""
Zone dataset v2: har TRADE ka result card + zyada "aankhein" (features).

v1 se farak:
  1. Multi-touch: ek zone pe 1st, 2nd, 3rd touch = alag-alag trade rows (touch_number).
     (Prajwal ka USDJPY: same zone, London open pe trade, phir NY open pe dobara.)
  2. H1 zones bhi (zone_tf = "H1"), M15 zones ke saath. Dono M5 pe simulate hote hain.
  3. Naye features:
       - Liquidity sweep : zone banne se pehle 6h / 24h low (bull) ya high (bear) sweep hua?
                           Asian range sweep? Previous day high/low sweep?
       - Premium/Discount: entry H4 (last 30 candles) aur D1 (last 20) range mein kahan hai (0 = bottom, 1 = top)
       - HTF bias        : H4 aur D1 ka aakhri BOS kis direction mein tha
       - Killzone        : fill London open (02-05 NY), NY open (07-10 NY), Asia open (19-22 NY) mein hua?
       - Displacement quality (Prajwal ke screenshots se): impulse_candles, impulse_body_pct, disp_d1,
                           followthrough_atr (zone banne ke baad price kitna door gaya)
       - Local trend     : trend_tf_aligned / trend_h1_aligned (HH/HL vs LH/LL swings, -6..+6; minus = trend ke ulta)
       - Approach        : approach_1h_atr / approach_3h_atr / approach_push (price zone mein kitna tez aaya)
  4. Labels: 3 exit rules ka R ek saath: r_trail (trailing 1R after 1R, best tha),
     r_tp1 (fixed 1:1), r_tp2 (fixed 1:2). `win` = r_trail > 0.

Touch rules:
  - Touch 1: zone bante hi body edge pe limit order, max LIFE_DAYS trading din tak.
  - Touch 2/3: pichla trade kam se kam 1R profit tak gaya ho (SL se pehle), phir price wapas entry pe aaye.
  - Zone khatam: koi M5 candle wick ke paar CLOSE ho, ya trade SL (-1R) pe gaya, ya life khatam.

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\research\zone_dataset_v2.py                       (3 core pairs, har pair ka displacement src/config.py se)
    python src\research\zone_dataset_v2.py --symbol EURUSD --disp 2.0   (manual override)

Output: data/research/{SYMBOL}_v2.parquet (har trade = ek row) + terminal pe summary.
LOOKAHEAD RULE: zone features sirf created_at tak ka data; fill features (time, killzone, touch)
sirf fill ke waqt tak ka. Labels (r_*) future se, wahi toh seekhna hai.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.features.smc import atr, find_orderflow_zones, find_swings, structure_breaks, track_zones  # noqa: E402
from src.research.management import POLICIES, run_trade  # noqa: E402
from src.research.zone_dataset import OUT, add_features, load, session_of  # noqa: E402
from src.features.volume_profile import add_vp_features  # noqa: E402
from src.features.activity import add_activity  # noqa: E402

NY = "America/New_York"
SL_BUFFER_ATR = 0.1
LIFE_DAYS = 5          # zone kitne trading din tak zinda (saare touches isi mein)
HOLD_DAYS = 3          # ek trade max kitne din
MAX_TOUCHES = 3
BARS_PER_DAY = 288     # M5
POL = {p.name: p for p in POLICIES}
LABEL_POLICIES = {"r_trail": POL["trail1"], "r_tp1": POL["tp1"], "r_tp2": POL["tp2"]}


# ----------------------------------------------------------------------
# Naye zone features (sirf created_at tak ka data)
# ----------------------------------------------------------------------
def asian_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Har NY date ki Asian range = pichli shaam 20:00 NY se raat 00:00 NY tak ka high/low."""
    ny = df.index.tz_convert(NY)
    m = (ny.hour >= 20)
    day = (ny[m] + pd.Timedelta(days=1)).date      # 20:00-23:59 wali candles agle din ki Asia range
    g = df.loc[m, ["high", "low"]].groupby(day)
    return pd.DataFrame({"asia_high": g["high"].max(), "asia_low": g["low"].min()})


class SwingTrend:
    """
    Local trend = pichle swings ka pattern (Prajwal: "market lower low bana raha tha").
    Aakhri 4 confirmed swing highs aur 4 swing lows dekhte hain -> 3+3 comparisons:
      HH (higher high) +1, HL (higher low) +1, LH (lower high) -1, LL (lower low) -1
    score +6 = strong uptrend, -6 = strong downtrend, 0 = range.
    Swing k tab confirm hota hai jab uske baad n candles ban jaayein (no lookahead).
    """

    def __init__(self, df: pd.DataFrame, n: int = 2):
        is_h, is_l = find_swings(df, n)
        self.n = n
        self.sh = np.nonzero(is_h)[0]
        self.sl = np.nonzero(is_l)[0]
        self.h = df["high"].to_numpy()
        self.l = df["low"].to_numpy()

    def score(self, i: int) -> float:
        """Candle i band hone tak confirmed swings ka trend score."""
        k_max = i - self.n
        if k_max < 0:
            return np.nan
        hs = self.sh[: np.searchsorted(self.sh, k_max, side="right")][-4:]
        ls = self.sl[: np.searchsorted(self.sl, k_max, side="right")][-4:]
        if len(hs) < 4 or len(ls) < 4:
            return np.nan
        dh = np.sign(np.diff(self.h[hs]))
        dl = np.sign(np.diff(self.l[ls]))
        return float(dh.sum() + dl.sum())


def zone_context(z: pd.DataFrame, df: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
                 bias_h4: pd.DataFrame, bias_d1: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    op, cl = df["open"].to_numpy(), df["close"].to_numpy()
    idx = df.index
    asia = asian_ranges(df)
    trend_tf = SwingTrend(df, 2)
    trend_h1 = SwingTrend(h1, 2)
    d1_atr = atr(d1).to_numpy()
    # candle ka size = AAM gap (median), pehli do candles ka gap nahi (weekend pe shuru ho toh galat aata tha;
    # live replay check, 2 Oct 2026). Backtest ke numbers same rehte hain (wahan bhi 15 min hi tha).
    step_min = pd.Series(idx[:500]).diff().median().total_seconds() / 60
    bars_6h = int(round(6 * 60 / step_min))
    bars_24h = 4 * bars_6h

    def last_closed(htf, t):
        return int(htf["close_time"].searchsorted(t, side="right")) - 1

    def last_bias(b, t):
        k = int(b["time"].searchsorted(t, side="right")) - 1
        return b["direction"].iloc[k] if k >= 0 else "none"

    rows = []
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        ob = int(idx.get_loc(r.zone_candle_time))
        i = int(idx.get_loc(r.bos_time))
        entry = r.body_top if bull else r.body_bottom

        # --- Liquidity sweep: zone ke aas-paas (ob-3 .. BOS) price ne purana low/high toda? ---
        w0 = max(ob - 3, 0)
        ext = lo[w0:i + 1].min() if bull else hi[w0:i + 1].max()

        def swept(n):
            a = max(w0 - n, 0)
            if a >= w0:
                return False
            ref = lo[a:w0].min() if bull else hi[a:w0].max()
            return bool(ext < ref) if bull else bool(ext > ref)

        ny_date = r.zone_candle_time.tz_convert(NY)
        ar = asia.loc[ny_date.date()] if (ny_date.hour < 17 and ny_date.date() in asia.index) else None
        swept_asia = False if ar is None else bool(ext < ar["asia_low"]) if bull else bool(ext > ar["asia_high"])

        k1 = last_closed(d1, r.created_at)
        swept_pd = False
        if k1 >= 0:
            swept_pd = bool(ext < d1["low"].iloc[k1]) if bull else bool(ext > d1["high"].iloc[k1])

        # --- Premium / discount: entry HTF range mein kahan hai ---
        def range_pos(htf, n):
            k = last_closed(htf, r.created_at)
            if k < n:
                return np.nan
            w = htf.iloc[k - n + 1:k + 1]
            top, bot = w["high"].max(), w["low"].min()
            return float((entry - bot) / (top - bot)) if top > bot else np.nan

        pos_h4, pos_d1 = range_pos(h4, 30), range_pos(d1, 20)
        b4, bd = last_bias(bias_h4, r.created_at), last_bias(bias_d1, r.created_at)

        # --- Displacement quality (Prajwal: "orderflow strong nahi tha, ek candle ke baad hi reverse") ---
        seg = slice(ob + 1, i + 1)                      # zone candle ke baad se BOS candle tak
        bodies = (cl[seg] - op[seg]) if bull else (op[seg] - cl[seg])
        rng = (hi[seg] - lo[seg]).sum()
        impulse_candles = int((bodies > 0).sum())       # direction mein band hui candles
        impulse_body_pct = float(bodies[bodies > 0].sum() / rng) if rng > 0 else np.nan
        move = (cl[i] - r.bottom) if bull else (r.top - cl[i])
        dk = last_closed(d1, r.created_at)
        disp_d1 = float(move / d1_atr[dk]) if dk >= 0 and d1_atr[dk] > 0 else np.nan

        # --- Local trend (swings) ---
        t_tf = trend_tf.score(i)
        k_h1 = last_closed(h1, r.created_at)
        t_h1 = trend_h1.score(k_h1) if k_h1 >= 0 else np.nan
        sgn = 1 if bull else -1

        rows.append(dict(
            sweep_6h=swept(bars_6h), sweep_24h=swept(bars_24h), sweep_asia=swept_asia, sweep_prev_day=swept_pd,
            range_pos_h4=pos_h4, range_pos_d1=pos_d1,
            # "discount" bull ke liye neeche (<0.5), bear ke liye upar (>0.5) = ICT wala sahi side
            pd_aligned_h4=(pos_h4 < 0.5) if bull else (pos_h4 > 0.5),
            pd_aligned_d1=(pos_d1 < 0.5) if bull else (pos_d1 > 0.5),
            bias_h4=b4, bias_d1=bd,
            bias_h4_aligned=b4 == r.direction, bias_d1_aligned=bd == r.direction,
            impulse_candles=impulse_candles, impulse_body_pct=impulse_body_pct, disp_d1=disp_d1,
            # trend "aligned": + = trend zone ke saath (buy in uptrend), - = ulta (buy in lower lows)
            trend_tf_aligned=t_tf * sgn, trend_h1_aligned=t_h1 * sgn,
        ))
    return pd.concat([z.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ----------------------------------------------------------------------
# Multi-touch trade simulation (M5)
# ----------------------------------------------------------------------
def killzone(ts: pd.Timestamp) -> str:
    h = ts.tz_convert(NY).hour
    if 2 <= h < 5:
        return "London open"
    if 7 <= h < 10:
        return "NY open"
    if 19 <= h < 22:
        return "Asia open"
    return "none"


def fill_context(bull: bool, entry: float, a: float, s: int, f: int,
                 hi: np.ndarray, lo: np.ndarray, cl: np.ndarray) -> dict:
    """
    Fill se PEHLE ki M5 candles (fill wali candle shaamil nahi) -> no lookahead.
      followthrough_atr : zone banne ke baad price zone se kitna door gaya (displacement aage chala?)
      approach_1h/3h_atr: fill se pehle 1h / 3h mein price zone ki taraf kitna tez aaya
                          (Prajwal: "sharp fall zone mein aaya -> sirf reaction milega, 1:1 lo")
      approach_push     : last 12 candles mein se kitni zone ki taraf band hui (0-1)
    """
    out = {}
    if f > s:
        far = hi[s:f].max() - entry if bull else entry - lo[s:f].min()
        out["followthrough_atr"] = float(far / a)
    else:
        out["followthrough_atr"] = 0.0
    for name, n in (("approach_1h_atr", 12), ("approach_3h_atr", 36)):
        k = max(f - n, 0)
        # bull: price upar se neeche zone mein aata hai -> (pehle ka close - entry)
        out[name] = float(((cl[k] - entry) if bull else (entry - cl[k])) / a) if f > k else 0.0
    k = max(f - 12, 1)
    steps = np.diff(cl[k - 1:f])
    toward = (steps < 0) if bull else (steps > 0)
    out["approach_push"] = float(toward.mean()) if len(steps) else np.nan
    return out


def simulate_touches(z: pd.DataFrame, m5: pd.DataFrame, entry_mode: str = "body") -> pd.DataFrame:
    """
    entry_mode:
      "body"        : limit exactly zone body edge pe. BUY tab bharta hai jab ASK wahan aaye.
      "body_spread" : BUY limit = body top + spread (zone bante waqt ka). Matlab jab chart ka BID
                      zone ko chhuye tabhi order bhar jaye ("exact tap" wale trades miss na hon).
                      SELL pe koi farak nahi (SELL limit waise bhi BID pe bharta hai = chart price).
    """
    lo, hi, cl, sp = (m5[c].to_numpy() for c in ["low", "high", "close", "spread"])
    idx = m5.index
    life, hold = LIFE_DAYS * BARS_PER_DAY, HOLD_DAYS * BARS_PER_DAY
    out = []

    for zi, r in enumerate(z.itertuples(index=False)):
        bull = r.direction == "bull"
        entry = r.body_top if bull else r.body_bottom
        if entry_mode == "body_spread" and bull:
            entry += r.spread_atr * r.atr      # spread at zone creation (pehle se pata hai, no lookahead)
        buf = SL_BUFFER_ATR * r.atr
        sl = (r.bottom - buf) if bull else (r.top + buf)
        risk = abs(entry - sl)
        s = int(idx.searchsorted(r.created_at))
        end = min(s + life, len(m5))
        if risk <= 0 or s >= end:
            continue

        # Zone dead: M5 close wick ke paar
        dead = (cl[s:end] < r.bottom) if bull else (cl[s:end] > r.top)
        # dead_at = zone marne wali candle ke BAAD wali candle. Marne wali candle mein bhi limit
        # fill ho sakta hai (price ek jhatke mein zone paar) -> wo trade bhi gina jaana chahiye.
        dead_at = s + int(np.argmax(dead)) + 1 if dead.any() else end
        # Fill possible: BUY limit ASK pe bharta hai, SELL limit BID pe
        can_fill = (lo[s:dead_at] + sp[s:dead_at] <= entry) if bull else (hi[s:dead_at] >= entry)

        pos = 0          # can_fill array mein kahan se dhoondhna hai
        touch = 0
        while touch < MAX_TOUCHES:
            hit = np.nonzero(can_fill[pos:])[0]
            if hit.size == 0:
                break
            f = s + pos + int(hit[0])
            touch += 1
            g = min(f + hold, len(m5))
            row = dict(zone_id=zi, touch_number=touch, fill_time=idx[f], entry=entry, sl=sl, risk=risk,
                       bars_since_created=f - s, killzone=killzone(idx[f]), session_fill=session_of(idx[f]),
                       fill_hour_ny=idx[f].tz_convert(NY).hour, fill_weekday=idx[f].tz_convert(NY).weekday())
            row.update(fill_context(bull, entry, r.atr, s, f, hi, lo, cl))
            for col, p in LABEL_POLICIES.items():
                rr, why = run_trade(p, bull, entry, risk, hi[f:g], lo[f:g], cl[f:g], sp[f:g])
                row[col] = round(float(rr), 3)
                row[col + "_exit"] = why
            out.append(row)

            # Agle touch ke liye: trade ne pehle 1R banaya ho, SL se pehle
            if bull:
                fav = (hi[f:dead_at] - entry) / risk
                adv = (lo[f:dead_at] - entry) / risk
            else:
                fav = (entry - (lo[f:dead_at] + sp[f:dead_at])) / risk
                adv = (entry - (hi[f:dead_at] + sp[f:dead_at])) / risk
            reach = np.nonzero(fav >= 1.0)[0]
            stop = np.nonzero(adv <= -1.0)[0]
            if reach.size == 0 or (stop.size and stop[0] <= reach[0]):
                break
            pos = (f - s) + int(reach[0]) + 1      # 1R ke baad wapas entry pe aaye = naya touch

    res = pd.DataFrame(out)
    return res


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def zones_for(tf_df, h1, h4, h1_zones, d1, bias_h4, bias_d1, swing_n, min_disp, zone_tf):
    z = track_zones(find_orderflow_zones(tf_df, swing_n=swing_n, min_disp_atr=min_disp), tf_df)
    z = add_features(z, tf_df, h1, h4, h1_zones)          # v1 wale features
    z = zone_context(z, tf_df, h4, d1, bias_h4, bias_d1, h1)   # v2 wale naye features
    z["zone_tf"] = zone_tf
    return z


def build(symbol: str, swing_n: int | None = None, min_disp: float | None = None, log=None,
          entry_mode: str = "body") -> pd.DataFrame:
    """swing_n / min_disp na do toh src/config.py se (har pair ka apna displacement)."""
    swing_n = config.SWING_N if swing_n is None else swing_n
    min_disp = config.disp_for(symbol) if min_disp is None else min_disp
    say = log or (lambda agent, msg, pct=None: print(f"[{agent}] {msg}"))
    say("zone", f"{symbol}: swing {swing_n}, min displacement {min_disp}x ATR", None)
    t0 = time.time()
    m5, m15, h1, h4, d1 = (load(symbol, tf) for tf in ["M5", "M15", "H1", "H4", "D1"])
    say("data", f"{symbol}: M5/M15/H1/H4/D1 loaded", 100)

    h1_zones = track_zones(find_orderflow_zones(h1, swing_n=swing_n, min_disp_atr=1.5), h1)
    bias_h4, bias_d1 = structure_breaks(h4, 2), structure_breaks(d1, 2)

    parts = []
    for tf, df in [("M15", m15), ("H1", h1)]:
        z = zones_for(df, h1, h4, h1_zones, d1, bias_h4, bias_d1, swing_n, min_disp, tf)
        say("zone", f"{symbol} {tf}: {len(z):,} zones + features", 50)
        t = simulate_touches(z, m5, entry_mode)
        z = z.reset_index(drop=True)
        t = t.merge(z, left_on="zone_id", right_index=True, how="left", suffixes=("", "_zone"))
        t["zone_id"] = tf + "_" + t["zone_id"].astype(str)
        parts.append(t)
        say("zone", f"{symbol} {tf}: {len(t):,} trades simulated", 100)

    ds = pd.concat(parts, ignore_index=True).sort_values("fill_time").reset_index(drop=True)
    ds["symbol"], ds["swing_n"], ds["min_disp"], ds["entry_mode"] = symbol, swing_n, min_disp, entry_mode
    ds["win"] = ds["r_trail"] > 0
    # Volume profile features (value area filter ke liye; src/features/volume_profile.py)
    ds = add_vp_features(ds, symbol, m5)
    # Market activity at the touch (activity filter; src/features/activity.py)
    ds = add_activity(ds, symbol, m5)
    say("zone", f"{symbol}: dataset v2 ready, {len(ds):,} trades in {time.time() - t0:.0f}s", 100)
    return ds


def save(ds: pd.DataFrame, symbol: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{symbol}_v2.parquet"
    ds.to_parquet(f)
    return f


# ----------------------------------------------------------------------
# Summary: har feature se PF kitna badalta hai
# ----------------------------------------------------------------------
def pf(r: pd.Series) -> float:
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 2) if gl else np.nan


def table(ds: pd.DataFrame, col: str) -> pd.DataFrame:
    g = ds.groupby(col, observed=True)
    return pd.DataFrame({
        "trades": g.size(),
        "win_%": g["r_trail"].apply(lambda r: round(100 * (r > 0).mean(), 1)),
        "PF_trail": g["r_trail"].apply(pf),
        "PF_1:1": g["r_tp1"].apply(pf),
        "PF_1:2": g["r_tp2"].apply(pf),
        "total_R_trail": g["r_trail"].sum().round(0),
    })


def summary(ds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = ds.copy()
    d["all"] = "all"
    d["range_h4"] = pd.cut(d["range_pos_h4"], [-0.01, 0.25, 0.5, 0.75, 1.01], labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    cols = ["all", "zone_tf", "touch_number", "killzone", "sweep_6h", "sweep_24h", "sweep_asia", "sweep_prev_day",
            "pd_aligned_h4", "pd_aligned_d1", "bias_h4_aligned", "bias_d1_aligned", "has_fvg", "h1_confluence"]
    return {c: table(d, c) for c in cols}


def main() -> None:
    p = argparse.ArgumentParser(description="Zone dataset v2 (multi-touch, H1 zones, sweep, PD, HTF bias, killzone)")
    p.add_argument("--symbol", nargs="+", default=config.CORE_SYMBOLS)
    p.add_argument("--swing", type=int, default=None, help="default: src/config.py (SWING_N)")
    p.add_argument("--disp", type=float, default=None, help="default: har pair ka apna (src/config.py DISP_BY_SYMBOL)")
    p.add_argument("--entry", choices=["body", "body_spread"], default="body",
                   help="body = limit at zone body edge; body_spread = BUY limit raised by the spread")
    a = p.parse_args()

    pd.set_option("display.width", 200)
    for sym in a.symbol:
        ds = build(sym, a.swing, a.disp, entry_mode=a.entry)
        print(f"saved -> {save(ds, sym)}\n")
        for name, t in summary(ds).items():
            print(f"== {sym}: {name} ==")
            print(t.to_string(), "\n")


if __name__ == "__main__":
    main()
