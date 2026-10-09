r"""
LIVE FEATURES: "abhi is 5-minute candle mein agar price zone ko chhuye, toh trade ka card kaisa dikhega?"

Backtest (zone_dataset_v2) ke BILKUL wahi functions use karta hai, taaki live = backtest:
  zones_for()      -> zone + zone features (sirf zone bante waqt tak ka data)
  fill_context()   -> approach / followthrough (sirf fill wali candle se PEHLE ki candles)
  add_vp_features()-> value area (fill se pehle ke 24h)
  add_activity()   -> market activity (last 15 min tick volume / last 24h, fill candle se pehle)
  cost features    -> walkforward.add_costs jaisa formula

Input: M5 candles (UTC, BID prices) jo ABHI TAK BAND ho chuki hain + abhi khuli candle ka open time.
Output: har "zinda, abhi tak na chhua" M15 zone ki ek row = ML ke liye ready card.

Bot sirf M15 zones ka PEHLA touch trade karta hai (backtest mein yahi best tha), isliye:
  - jis zone ko pehle hi chhua ja chuka hai (touch 1 ho gaya) -> skip
  - zone dead (M5 close wick ke paar) ya 5 trading din purana -> skip
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data.build_timeframes import resample, trading_clock  # noqa: E402
from src.features.smc import find_orderflow_zones, structure_breaks, track_zones  # noqa: E402
from src.features.activity import add_activity  # noqa: E402
from src.live import spread_floor as sfloor  # noqa: E402
from src.features.volume_profile import add_vp_features  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402
from src.research.zone_dataset import add_features, session_of  # noqa: E402
from src.research.zone_dataset_v2 import (BARS_PER_DAY, LIFE_DAYS, SL_BUFFER_ATR, fill_context,  # noqa: E402
                                          killzone, zone_context)

NY = "America/New_York"
HTF_RULES = {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "D"}
HISTORY_DAYS = 200          # kitna purana data live mein chahiye (ATR / D1 range / bias warm-up)
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}


def closed_only(htf: pd.DataFrame, rule: str, now: pd.Timestamp) -> pd.DataFrame:
    """Bade timeframe ki aakhri candle abhi bani ho rahi ho toh hata do (backtest mein sirf BAND candles thi)."""
    if htf.empty:
        return htf
    start_clock = trading_clock(pd.DatetimeIndex([htf.index[-1]]))[0]
    end_clock = start_clock.floor(rule) + pd.Timedelta(rule if rule != "D" else "1D")
    now_clock = trading_clock(pd.DatetimeIndex([now]))[0]
    return htf if now_clock >= end_clock else htf.iloc[:-1]


def frames(m5: pd.DataFrame, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """M5 (band candles) -> M15 / H1 / H4 / D1, sirf poori bani candles."""
    out = {"M5": m5}
    for name, rule in HTF_RULES.items():
        out[name] = closed_only(resample(m5, rule), rule, now)
    return out


def prepare(d: pd.DataFrame, commission: float = wf.COMMISSION) -> pd.DataFrame:
    """walkforward.load() wale derived features (training jaisa hi), bina labels ke."""
    d = d.copy()
    d["spread_r"] = d["spread_atr"] * d["atr"] / d["risk"]
    d["risk_atr"] = d["risk"] / d["atr"]
    pip = d["symbol"].astype(str).map(wf.PIP)
    usd_quote = d["symbol"].astype(str).str.endswith("USD")
    per_lot = np.where(usd_quote, d["risk"] * 100_000, d["risk"] * 100_000 / d["entry"])
    d["cost_r"] = commission / per_lot
    d["sl_pips"] = d["risk"] / pip
    for c in wf.BOOL:
        d[c] = d[c].astype(float)
    for c in wf.CATEGORICAL:
        d[c] = d[c].astype(str).astype("category")
    return d


def candidates(symbol: str, m5: pd.DataFrame, bar_time: pd.Timestamp,
               zone_lookback_days: int = LIFE_DAYS + 3, spread_floor: bool | None = None) -> pd.DataFrame:
    """
    m5       : band M5 candles (UTC index, columns open/high/low/close/volume/spread), aakhri = abhi wali se pehle
    bar_time : abhi khuli M5 candle ka open time (= backtest ka fill_time agar is candle mein fill ho)
    spread_floor : True = ML features ke liye spread ko training jaisa rakho (src/live/spread_floor.py).
                   None = config.ML_SPREAD_FLOOR. Replay (Dukascopy data) mein False, taaki backtest se match ho.
    Returns  : har eligible M15 zone ki row (features + entry / sl / risk), ML ke liye ready.
               spread_live / spread_used = zone bante waqt ka asli vs ML ko diya gaya spread (price units).
    """
    if "close_time" not in m5:
        m5 = m5.assign(close_time=m5.index + pd.Timedelta(minutes=5), m5_count=1)
    if spread_floor is None:
        spread_floor = config.ML_SPREAD_FLOOR
    m5_ml = sfloor.apply(m5, symbol) if spread_floor else m5     # sirf ML features ke liye
    fr = frames(m5_ml, bar_time)
    m15, h1, h4, d1 = fr["M15"], fr["H1"], fr["H4"], fr["D1"]
    swing, disp = config.SWING_N, config.disp_for(symbol)

    h1_zones = track_zones(find_orderflow_zones(h1, swing_n=swing, min_disp_atr=1.5), h1)
    bias_h4, bias_d1 = structure_breaks(h4, 2), structure_breaks(d1, 2)
    z = find_orderflow_zones(m15, swing_n=swing, min_disp_atr=disp)
    if z.empty:
        return pd.DataFrame()
    z = z[z["created_at"] >= bar_time - pd.Timedelta(days=zone_lookback_days)]   # sirf haal ke zones
    z = z[z["created_at"] <= bar_time]
    if z.empty:
        return pd.DataFrame()
    z = track_zones(z.reset_index(drop=True), m15)
    z = add_features(z, m15, h1, h4, h1_zones)
    z = zone_context(z, m15, h4, d1, bias_h4, bias_d1, h1)
    z["zone_tf"] = "M15"

    lo, hi, cl, sp = (m5[c].to_numpy() for c in ["low", "high", "close", "spread"])   # ASLI spread (fill check)
    idx = m5.index
    real_m15_spread = m5["spread"].resample("15min").mean()      # zone bante waqt ka asli spread (log ke liye)
    f = len(m5)                                  # abhi wali candle ka index (backtest jaisa: array mein agla)
    life = LIFE_DAYS * BARS_PER_DAY
    rows = []
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        entry = config.entry_price(r, bull)
        buf = SL_BUFFER_ATR * r.atr
        sl = (r.bottom - buf) if bull else (r.top + buf)
        risk = abs(entry - sl)
        s = int(idx.searchsorted(r.created_at))
        if risk <= 0 or s > f or f - s >= life:
            continue
        # zone abhi tak zinda? (koi band M5 candle wick ke paar CLOSE nahi hui)
        dead = (cl[s:f] < r.bottom) if bull else (cl[s:f] > r.top)
        if dead.any():
            continue
        # pehla touch abhi tak nahi hua? (BUY limit ASK pe bharta, SELL limit BID pe)
        touched = (lo[s:f] + sp[s:f] <= entry) if bull else (hi[s:f] >= entry)
        if touched.any():
            continue
        row = r._asdict()
        row.update(symbol=symbol, touch_number=1, fill_time=bar_time, entry=entry, sl=sl, risk=risk,
                   bars_since_created=f - s, killzone=killzone(bar_time), session_fill=session_of(bar_time),
                   fill_hour_ny=bar_time.tz_convert(NY).hour, fill_weekday=bar_time.tz_convert(NY).weekday(),
                   spread_live=float(real_m15_spread.get(r.bos_time, np.nan)),
                   spread_used=float(r.spread_atr * r.atr))
        row.update(fill_context(bull, entry, r.atr, s, f, hi, lo, cl))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d = add_vp_features(d, symbol, m5)
    d["va_ok"] = config.va_ok(d["r24_va_pos"])
    d = add_activity(d, symbol, m5)
    d["act_ok"] = config.act_ok(d["act_15m"])
    return prepare(d)
