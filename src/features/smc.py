"""
SMC (Smart Money Concepts) detectors - v1

Is file mein wo "aankhen" hain jo chart pe SMC cheezein dhoondhti hain:
  - atr()                 : volatility (ek candle normally kitna hilti hai)
  - find_swings()         : swing highs / lows (structure ka base)
  - find_orderflow_zones(): BOS + displacement + orderflow zone (SMC mein isko Order Block bhi kehte hain)
  - track_zones()         : zone bante hi aage kya hua (touch, kitna ghusa, kab toota)

LOOKAHEAD RULE (bahut important):
  Har cheez sirf us waqt tak ke data se banti hai jo us waqt pata tha.
  - Swing high tab confirm hota hai jab uske baad `swing_n` candles ban jayein.
  - Zone tab "exist" karta hai jab BOS wali candle BAND ho jaye (created_at).
  Backtest mein zone ko created_at se pehle kabhi use nahi karna.

Settings (swing_n, min_disp_atr, ...) jaan-boojh ke parameters hain, taaki
backtest mein alag values try karke data se best nikal sakein.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# ATR: Average True Range
# ----------------------------------------------------------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    True Range = in teeno mein se sabse bada:
        high - low, |high - pichla close|, |low - pichla close|
    ATR = True Range ka smooth average (Wilder method).
    Isse "bada move" ko pair aur time ke hisaab se measure kar sakte hain:
    gold 2016 mein $5 hilta tha, 2026 mein $20, lekin "2x ATR" dono mein bada move hai.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------------
# Swings
# ----------------------------------------------------------------------
def find_swings(df: pd.DataFrame, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Swing high = wo candle jiska high, dono taraf ki n candles se ooncha ho.
    Swing low  = wo candle jiska low, dono taraf ki n candles se neecha ho.

    Returns: (is_swing_high, is_swing_low) boolean arrays.
    DHYAN: candle i ka swing hona candle i+n pe hi pata chalta hai (right side ki n candles chahiye).
    """
    window = 2 * n + 1
    is_high = (df["high"] == df["high"].rolling(window, center=True).max()).to_numpy()
    is_low = (df["low"] == df["low"].rolling(window, center=True).min()).to_numpy()
    return is_high, is_low


# ----------------------------------------------------------------------
# Orderflow zones (Order Blocks)
# ----------------------------------------------------------------------
def find_orderflow_zones(
    df: pd.DataFrame,
    swing_n: int = 3,
    min_disp_atr: float = 2.0,
    max_ob_lookback: int = 10,
    atr_n: int = 14,
) -> pd.DataFrame:
    """
    Orderflow zone (Order Block) dhoondhta hai. Logic (bullish, bearish ulta):

      1. BOS: candle ka CLOSE pichle confirmed swing high ke upar.
         (har swing high ek hi baar "break" hota hai)
      2. Zone candle: BOS candle se peeche jaate hue pehli RED candle
         = "displacement se pehle ki aakhri red candle".
      3. Displacement strength = (BOS close - zone candle low) / ATR.
         min_disp_atr se kam hai toh move kamzor tha -> zone nahi banega.
      4. Zone = zone candle ka range. Dono version save karte hain:
         wick zone (high-low) aur body zone (open-close). Kaunsa better hai,
         ye backtest batayega.

    Returns: har zone ki ek row.
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    a = atr(df, atr_n).to_numpy()
    times = df.index
    close_times = df["close_time"].to_numpy() if "close_time" in df else times
    is_sh, is_sl = find_swings(df, swing_n)

    last_sh = last_sl = -1
    sh_broken = sl_broken = True
    zones = []

    for i in range(len(df)):
        # Bar (i - swing_n) ka swing ab confirm hua (uske baad swing_n candles ban gayi)
        k = i - swing_n
        if k >= 0:
            if is_sh[k]:
                last_sh, sh_broken = k, False
            if is_sl[k]:
                last_sl, sl_broken = k, False

        # ---------------- Bullish BOS ----------------
        if not sh_broken and last_sh >= 0 and c[i] > h[last_sh]:
            sh_broken = True
            ob = _find_opposite_candle(o, c, i, max_ob_lookback, bullish=True)
            if ob is not None and a[ob] > 0:
                disp = (c[i] - l[ob]) / a[ob]
                if disp >= min_disp_atr:
                    zones.append(_zone_row("bull", ob, i, last_sh, disp, o, h, l, c, a, times, close_times))

        # ---------------- Bearish BOS ----------------
        if not sl_broken and last_sl >= 0 and c[i] < l[last_sl]:
            sl_broken = True
            ob = _find_opposite_candle(o, c, i, max_ob_lookback, bullish=False)
            if ob is not None and a[ob] > 0:
                disp = (h[ob] - c[i]) / a[ob]
                if disp >= min_disp_atr:
                    zones.append(_zone_row("bear", ob, i, last_sl, disp, o, h, l, c, a, times, close_times))

    return pd.DataFrame(zones)


def _find_opposite_candle(o, c, i, lookback, bullish):
    """BOS candle i se peeche jaake pehli ulti candle (bullish move ke liye red, bearish ke liye green)."""
    for j in range(i - 1, max(i - 1 - lookback, -1), -1):
        if bullish and c[j] < o[j]:
            return j
        if not bullish and c[j] > o[j]:
            return j
    return None


def _zone_row(direction, ob, i, swing_idx, disp, o, h, l, c, a, times, close_times):
    # FVG check: displacement ke beech koi fair value gap bana? (3-candle gap)
    has_fvg = False
    for k in range(ob + 1, i):
        if k + 1 > i:
            break
        if direction == "bull" and l[k + 1] > h[k - 1]:
            has_fvg = True
            break
        if direction == "bear" and h[k + 1] < l[k - 1]:
            has_fvg = True
            break

    return dict(
        direction=direction,
        zone_candle_time=times[ob],            # zone candle kab khuli
        bos_time=times[i],                     # BOS candle kab khuli
        created_at=close_times[i],             # zone kab se "exist" karta hai (BOS candle band hone pe)
        top=h[ob], bottom=l[ob],               # wick zone
        body_top=max(o[ob], c[ob]), body_bottom=min(o[ob], c[ob]),   # body zone
        broken_swing=h[swing_idx] if direction == "bull" else l[swing_idx],
        disp_atr=round(float(disp), 2),        # displacement kitna strong tha (ATR mein)
        atr=a[ob],
        bars_ob_to_bos=i - ob,
        has_fvg=has_fvg,
    )


# ----------------------------------------------------------------------
# Zone ke baad kya hua
# ----------------------------------------------------------------------
def track_zones(zones: pd.DataFrame, df: pd.DataFrame, use_body: bool = False, max_bars: int = 500) -> pd.DataFrame:
    """
    Har zone ke liye, created_at ke BAAD ki candles dekh ke:
      - first_touch_time      : price pehli baar zone mein kab aaya
      - first_touch_depth     : pehle touch mein zone ke kitna andar gaya (0 = sirf chhua, 1 = poora zone, >1 = paar)
      - invalidated_time      : kab candle zone ke paar CLOSE hui (zone dead)
      - touches               : invalid hone se pehle kitni baar zone mein aaya
    Ye sab baad mein ML ke features / labels banenge.
    """
    top_col, bot_col = ("body_top", "body_bottom") if use_body else ("top", "bottom")
    lows, highs, closes = df["low"].to_numpy(), df["high"].to_numpy(), df["close"].to_numpy()
    idx = df.index
    out = []

    for z in zones.itertuples(index=False):
        top, bottom = getattr(z, top_col), getattr(z, bot_col)
        height = max(top - bottom, 1e-9)
        start = idx.searchsorted(z.created_at)          # created_at ke baad wali pehli candle
        end = min(start + max_bars, len(df))
        lo, hi, cl = lows[start:end], highs[start:end], closes[start:end]

        if z.direction == "bull":
            inside = lo <= top
            depth = (top - lo) / height
            dead = cl < bottom
        else:
            inside = hi >= bottom
            depth = (hi - bottom) / height
            dead = cl > top

        dead_pos = np.argmax(dead) if dead.any() else None
        limit = dead_pos + 1 if dead_pos is not None else len(inside)
        ins = inside[:limit]
        entries = ins & ~np.concatenate([[False], ins[:-1]])   # bahar se andar aaya = naya touch

        first = np.argmax(ins) if ins.any() else None
        out.append(dict(
            first_touch_time=idx[start + first] if first is not None else pd.NaT,
            first_touch_depth=round(float(depth[first]), 2) if first is not None else np.nan,
            invalidated_time=idx[start + dead_pos] if dead_pos is not None else pd.NaT,
            touches=int(entries.sum()),
        ))

    result = pd.DataFrame(out, columns=["first_touch_time", "first_touch_depth", "invalidated_time", "touches"])
    # Agar kisi column mein saare NaT hon toh pandas timezone bhool jata hai -> hamesha UTC set karo
    for col in ["first_touch_time", "invalidated_time"]:
        result[col] = pd.to_datetime(result[col], utc=True)
    return pd.concat([zones.reset_index(drop=True), result], axis=1)
