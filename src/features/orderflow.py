r"""
REAL ORDERFLOW features from CME futures trades (Databento).

Raw data (src/data/databento_download.py): data/orderflow/raw/{6E|6J|6B}_{YYYY-MM}.parquet
  every futures trade: ts_event, price, size, side (B = buyer aggressive, A = seller aggressive)

STEP 1 - build_bars(): 1-minute orderflow bars per future -> data/orderflow/bars/{FUT}_1m.parquet
  buy, sell        : aggressive buy / sell volume (contracts)
  big_buy, big_sell: same, only trades with size >= BIG (20 contracts ~ top 2% of trades, "institution size")
  n                : number of trades
  close            : last traded price in the minute
  inst             : instrument id (contract month; changes at roll every ~3 months)

STEP 2 - add_orderflow_features(): per trade, ONLY data before the fill bar (no lookahead).
  Everything is DIRECTION-ADJUSTED: positive = flow in the trade's direction
  (bull trade: buyers aggressive = positive). 6J = JPY/USD = opposite of USDJPY, so its sign is flipped.

  of_disp_delta     : (buy-sell)/(buy+sell) during the displacement (zone candle close -> BOS candle close)
                      = was the move that created the zone backed by real aggressive orders?
  of_disp_vol       : volume per minute in the displacement / avg volume per minute of last 24h
  of_disp_big       : big-trade delta in the displacement (big buy - big sell) / big total
  of_zone_delta     : delta inside the zone candle itself
  of_delta_15m      : delta in the last 15 min before the fill (price coming INTO the zone)
  of_delta_60m      : same, last 60 min
  of_big_delta_60m  : big-trade delta, last 60 min
  of_vol_15m        : volume per minute last 15 min / avg of last 24h (is the market busy at the touch?)
  of_ret_60m_atr    : futures price move last 60 min, in trade direction, in ATR (spot ATR of the zone tf)
  of_absorb_60m     : -of_delta_60m - of_ret_60m_atr ... see below

  ABSORPTION idea: price falls into a demand zone (sellers aggressive, delta negative) but price
  does NOT fall as much as that selling "should" push it -> somebody big is absorbing. We give the ML
  both raw parts (delta and price move) + this simple combination:
      of_absorb_60m = (-of_delta_60m) * 10 - max(-of_ret_60m_atr, 0)
  high = strong selling against the trade but little price damage.

Missing data (trades before 2024, or a minute with no futures trades) -> NaN. LightGBM handles NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "orderflow" / "raw"
BARS = ROOT / "data" / "orderflow" / "bars"
BIG = 20
FUT_FOR = {"EURUSD": ("6E", 1), "GBPUSD": ("6B", 1), "USDJPY": ("6J", -1)}   # (future, sign vs spot pair)
TF_MIN = {"M15": 15, "H1": 60, "H4": 240}
OF_FEATURES = ["of_disp_delta", "of_disp_vol", "of_disp_big", "of_zone_delta", "of_delta_15m",
               "of_delta_60m", "of_big_delta_60m", "of_vol_15m", "of_ret_60m_atr", "of_absorb_60m"]


# ----------------------------------------------------------------------
# STEP 1: raw trades -> 1-minute bars
# ----------------------------------------------------------------------
def bars_from_trades(t: pd.DataFrame) -> pd.DataFrame:
    t = t[t["side"].isin(["B", "A"])]
    m = t["ts_event"].dt.floor("min")
    buy = t["size"].where(t["side"] == "B", 0)
    sell = t["size"].where(t["side"] == "A", 0)
    big = t["size"] >= BIG
    g = pd.DataFrame({"m": m, "buy": buy, "sell": sell, "big_buy": buy.where(big, 0),
                      "big_sell": sell.where(big, 0), "price": t["price"],
                      "inst": t["instrument_id"]}).groupby("m")
    out = g[["buy", "sell", "big_buy", "big_sell"]].sum()
    out["n"] = g.size()
    out["close"] = g["price"].last()
    out["inst"] = g["inst"].last()
    out.index.name = "time"
    return out.astype({"buy": "int64", "sell": "int64", "big_buy": "int64", "big_sell": "int64", "n": "int32"})


def build_bars(fut: str, log=print) -> pd.DataFrame | None:
    files = sorted(RAW.glob(f"{fut}_*.parquet"))
    if not files:
        return None
    BARS.mkdir(parents=True, exist_ok=True)
    out_file = BARS / f"{fut}_1m.parquet"
    parts = [bars_from_trades(pd.read_parquet(f)) for f in files]
    b = pd.concat(parts).sort_index()
    b = b[~b.index.duplicated(keep="last")]
    b.to_parquet(out_file)
    log(f"{fut}: {len(files)} months -> {len(b):,} one-minute bars ({b.index.min():%Y-%m-%d} -> {b.index.max():%Y-%m-%d})")
    return b


def load_bars(fut: str) -> pd.DataFrame | None:
    f = BARS / f"{fut}_1m.parquet"
    return pd.read_parquet(f) if f.exists() else None


# ----------------------------------------------------------------------
# STEP 2: features per trade
# ----------------------------------------------------------------------
class _Window:
    """Fast sums over [a, b) time windows using cumulative sums on the minute bars."""

    def __init__(self, b: pd.DataFrame):
        self.t = b.index.values.astype("datetime64[ns]")
        z = lambda c: np.concatenate([[0], np.cumsum(b[c].to_numpy(np.float64))])  # noqa: E731
        self.buy, self.sell, self.bb, self.bs = z("buy"), z("sell"), z("big_buy"), z("big_sell")
        self.close = b["close"].to_numpy(np.float64)
        self.inst = b["inst"].to_numpy()

    def idx(self, ts) -> np.ndarray:
        return np.searchsorted(self.t, np.asarray(ts, dtype="datetime64[ns]"), side="left")

    def sums(self, a, b):
        ia, ib = self.idx(a), self.idx(b)
        return (self.buy[ib] - self.buy[ia], self.sell[ib] - self.sell[ia],
                self.bb[ib] - self.bb[ia], self.bs[ib] - self.bs[ia], ia, ib)


def _norm_delta(buy, sell):
    tot = buy + sell
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, (buy - sell) / tot, np.nan)


def add_orderflow_features(d: pd.DataFrame, symbol: str, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """d = zone_dataset_v2 rows of ONE symbol. Adds OF_FEATURES (NaN where no futures data)."""
    d = d.copy()
    for c in OF_FEATURES:
        d[c] = np.nan
    if symbol not in FUT_FOR:
        return d
    fut, sign = FUT_FOR[symbol]
    b = bars if bars is not None else load_bars(fut)
    if b is None or b.empty:
        return d
    w = _Window(b)
    first, last = b.index.min(), b.index.max()
    ok = (d["fill_time"] >= first + pd.Timedelta(days=1)) & (d["fill_time"] <= last)
    if not ok.any():
        return d
    x = d.loc[ok]
    dirn = np.where(x["direction"].astype(str) == "bull", 1.0, -1.0) * sign   # +1 = buyer aggression helps
    tf = pd.to_timedelta(x["zone_tf"].astype(str).map(TF_MIN).fillna(15).to_numpy(), unit="m")
    to_utc = lambda s: pd.DatetimeIndex(s).tz_convert("UTC").tz_localize(None).values  # noqa: E731
    zc = to_utc(x["zone_candle_time"])
    zone_end = zc + tf.values
    bos_end = to_utc(x["bos_time"]) + tf.values
    fill = to_utc(x["fill_time"])
    h = np.timedelta64(1, "h")
    m15 = np.timedelta64(15, "m")

    # 24h baseline volume per ACTIVE minute (minutes with trades; weekends / breaks don't dilute it)
    bb, ss, _, _, ja, jb = w.sums(fill - 24 * h, fill)
    base = np.where(jb - ja >= 60, (bb + ss) / np.maximum(jb - ja, 1), np.nan)

    # displacement
    db, ds, dbb, dbs, ia, ib = w.sums(zone_end, bos_end)
    mins = np.maximum(ib - ia, 1)                       # active minutes in the displacement
    with np.errstate(divide="ignore", invalid="ignore"):
        disp_vol = np.where(base > 0, (db + ds) / mins / base, np.nan)
    disp_delta = _norm_delta(db, ds) * dirn
    disp_big = _norm_delta(dbb, dbs) * dirn

    # zone candle
    zb, zs, *_ = w.sums(zc, zone_end)
    zone_delta = _norm_delta(zb, zs) * dirn

    # approach into the zone
    b15, s15, _, _, k15a, k15b = w.sums(fill - m15, fill)
    b60, s60, bb60, bs60, i60a, i60b = w.sums(fill - h, fill)
    delta15 = _norm_delta(b15, s15) * dirn
    delta60 = _norm_delta(b60, s60) * dirn
    big60 = _norm_delta(bb60, bs60) * dirn
    with np.errstate(divide="ignore", invalid="ignore"):
        vol15 = np.where(base > 0, (b15 + s15) / np.maximum(k15b - k15a, 1) / base, np.nan)

    # futures price move last 60 min (same contract only), in trade direction, in ATR
    n = len(w.close)
    ia_c = np.clip(i60a, 0, n - 1)
    ib_c = np.clip(i60b - 1, 0, n - 1)
    same = (w.inst[ia_c] == w.inst[ib_c]) & (i60b > i60a)
    logret = np.log(w.close[ib_c] / w.close[ia_c])
    spot = x["entry"].to_numpy(float)
    atr = x["atr"].to_numpy(float)
    ret_atr = np.where(same, logret * sign * spot / atr, np.nan) * np.where(x["direction"].astype(str) == "bull", 1, -1)
    absorb = (-delta60) * 10 - np.maximum(-ret_atr, 0)

    vals = dict(of_disp_delta=disp_delta, of_disp_vol=disp_vol, of_disp_big=disp_big, of_zone_delta=zone_delta,
                of_delta_15m=delta15, of_delta_60m=delta60, of_big_delta_60m=big60, of_vol_15m=vol15,
                of_ret_60m_atr=ret_atr, of_absorb_60m=absorb)
    for k, v in vals.items():
        d.loc[ok, k] = v
    return d


if __name__ == "__main__":
    import sys
    futs = sys.argv[1:] or ["6E", "6J", "6B"]
    for f in futs:
        build_bars(f)
