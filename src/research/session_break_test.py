r"""
TEST: "Displacement ne PICHLE SESSION ka high/low toda ya nahi?"  (Prajwal ka idea, 23-24 Sep USDJPY loss)

Uska losing sell: Asia open pe girawat aayi (zone bana), lekin us girawat ne pichle NY session ka LOW
nahi toda -> chhota internal move tha -> London mein price wapas upar, SL.

FEATURE (pehle se fix, koi tuning nahi):
  Sessions (New York time):  Asia 19:00-02:00,  London 02:00-08:00,  New York 08:00-17:00
  prev session   = zone candle se pehle POORA ho chuka aakhri session
  disp extreme   = zone candle se BOS candle close tak ka lowest low (sell zone) / highest high (buy zone)
  sess_break     = True agar displacement ne prev session ka low (sell) / high (buy) toda
  sess_beyond_atr= kitna aage gaya (+) ya kitna peeche reh gaya (-), ATR mein

PASS RULES (pehle se tay):
  - effect 2021 se pehle aur baad DONO halves mein same direction
  - teeno pairs pe same direction
  - ML ya rule se bot PF sudhaar noise (~0.02) se saaf zyada
  Warna feature drop.

Run: python src\research\session_break_test.py   -> data/research/session_break_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402

DATA = ROOT / "data" / "processed"
RESEARCH = ROOT / "data" / "research"
NY = "America/New_York"
TF_MIN = {"M15": 15, "H1": 60}


def sessions(m5: pd.DataFrame) -> pd.DataFrame:
    """One row per session: start, end (UTC), high, low."""
    ny = m5.index.tz_convert(NY)
    h = ny.hour
    name = np.where((h >= 19) | (h < 2), "asia", np.where(h < 8, "london", np.where(h < 17, "ny", "off")))
    day = (ny + pd.Timedelta(hours=5)).normalize()          # asia starting 19:00 belongs to next day
    sid = pd.Series(name, index=m5.index) + "_" + pd.Series(day.strftime("%Y-%m-%d"), index=m5.index)
    df = pd.DataFrame({"sid": sid.values, "name": name, "high": m5["high"].values, "low": m5["low"].values,
                       "t": m5.index}, index=m5.index)
    df = df[df["name"] != "off"]
    g = df.groupby("sid", sort=False)
    s = pd.DataFrame({"start": g["t"].min(), "end": g["t"].max() + pd.Timedelta(minutes=5),
                      "high": g["high"].max(), "low": g["low"].min()})
    return s.sort_values("end").reset_index(drop=True)


def add_session_break(d: pd.DataFrame, symbol: str) -> pd.DataFrame:
    m5 = pd.read_parquet(DATA / f"{symbol}_M5.parquet", columns=["high", "low"])
    s = sessions(m5)
    ends = s["end"].values
    hi, lo = m5["high"].to_numpy(), m5["low"].to_numpy()
    idx = m5.index
    d = d.copy()
    tf = pd.to_timedelta(d["zone_tf"].astype(str).map(TF_MIN).fillna(15).to_numpy(), unit="m")
    zc = pd.DatetimeIndex(d["zone_candle_time"])
    bos_end = pd.DatetimeIndex(d["bos_time"]) + tf
    k = np.searchsorted(ends, zc.values, side="right") - 1          # last session ended at/before zone candle
    a = idx.searchsorted(zc)
    b = idx.searchsorted(bos_end)
    bull = d["direction"].astype(str).to_numpy() == "bull"
    brk = np.full(len(d), np.nan)
    beyond = np.full(len(d), np.nan)
    atr = d["atr"].to_numpy(float)
    for i in range(len(d)):
        if k[i] < 0 or b[i] <= a[i]:
            continue
        if bull[i]:
            ext = hi[a[i]:b[i]].max()
            dist = ext - s["high"].iat[k[i]]
        else:
            ext = lo[a[i]:b[i]].min()
            dist = s["low"].iat[k[i]] - ext
        brk[i] = float(dist > 0)
        beyond[i] = dist / atr[i]
    d["sess_break"] = brk
    d["sess_beyond_atr"] = beyond
    return d


def pf(r):
    r = pd.Series(r).dropna()
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else None


def split(df, col, rcol):
    out = {}
    for v, g in df.groupby(df[col]):
        out[str(int(v))] = dict(trades=len(g), pf=pf(g[rcol]), early=pf(g.loc[g["year"] < 2021, rcol]),
                                late=pf(g.loc[g["year"] >= 2021, rcol]),
                                by_pair={s: pf(x[rcol]) for s, x in g.groupby(g["symbol"].astype(str))})
    return out


def run(symbols=None, log=print) -> dict:
    symbols = symbols or config.CORE_SYMBOLS
    d = wf.load(symbols)
    d = pd.concat([add_session_break(g, s) for s, g in d.groupby(d["symbol"].astype(str))])
    d = d.sort_values("fill_time").reset_index(drop=True)
    d["year"] = d["fill_time"].dt.year
    res = {"share_broke": round(float(d["sess_break"].mean()), 3)}

    # 1) all M15 first-touch trades (no ML), net trailing R
    m = d[(d["zone_tf"].astype(str) == "M15") & (d["touch_number"] == 1)]
    res["all_m15_touch1"] = split(m, "sess_break", "r_trail")

    # 2) inside the bot's trades (base ML, walk-forward 2019+)
    base = list(wf.FEATURES)
    w = wf.walk_forward(d, 2019, log=lambda *a: None)
    b = w[(w["zone_tf"].astype(str) == "M15") & (w["touch_number"] == 1) & w["keep_0.3"] & w["va_ok"]].copy()
    b["r"] = np.where(b["exit_ml"] == "trail", b["r_trail"], b["r_tp1"])
    res["bot_base"] = dict(trades=len(b), pf=pf(b["r"]), total_r=round(float(b["r"].sum()), 1))
    res["bot_split"] = split(b, "sess_break", "r")

    # 3) rule: skip "not broke" if it was worse BEFORE 2019 (decided on old data only)
    pre = m[m["year"] < 2019]
    bad = 0 if (pf(pre.loc[pre["sess_break"] == 0, "r_trail"]) or 9) < (pf(pre.loc[pre["sess_break"] == 1, "r_trail"]) or 9) else 1
    keep = b["sess_break"] != bad
    rr = b.loc[keep, "r"]
    res["rule"] = dict(skip=("not broke" if bad == 0 else "broke"), trades=int(keep.sum()), pf=pf(rr),
                       total_r=round(float(rr.sum()), 1),
                       by_pair={s: pf(rr[b.loc[keep, "symbol"].astype(str) == s]) for s in symbols},
                       early=pf(b.loc[keep & (b["year"] < 2022), "r"]), late=pf(b.loc[keep & (b["year"] >= 2022), "r"]),
                       base_early=pf(b.loc[b["year"] < 2022, "r"]), base_late=pf(b.loc[b["year"] >= 2022, "r"]))

    # 4) ML with the 2 new features
    wf.FEATURES = base + ["sess_break", "sess_beyond_atr"]
    try:
        w2 = wf.walk_forward(d, 2019, log=lambda *a: None)
    finally:
        wf.FEATURES = base
    b2 = w2[(w2["zone_tf"].astype(str) == "M15") & (w2["touch_number"] == 1) & w2["keep_0.3"] & w2["va_ok"]]
    r2 = np.where(b2["exit_ml"] == "trail", b2["r_trail"], b2["r_tp1"])
    res["bot_ml_with_feature"] = dict(trades=len(b2), pf=pf(r2), total_r=round(float(r2.sum()), 1),
                                      by_pair={s: pf(r2[(b2["symbol"].astype(str) == s).to_numpy()]) for s in symbols})
    (RESEARCH / "session_break_test.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
