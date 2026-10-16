"""
Confluence test for the 15m zone bot (16 Oct 2026). Prajwal: "koi bhi confluence, sab try karke dekho".

RULES (run se PEHLE fix). Har confluence = fill ke waqt pata (no lookahead), M15 zone, touch 1.
  NEW:
    h4_zone      M15 zone same direction ke active H4 zone (BOS swing 5, untapped-ya-zinda, 60 d) se overlap kare
    pd_level     bull: pichle trading din ka LOW zone ke andar (+-0.3 ATR); bear: pichle din ka HIGH
    pw_level     same, pichle HAFTE ka low / high
    asia_level   same, aaj ki Asia range low / high (fill 00:00 NY ke baad)
    round_50     00 / 50 level (EURUSD/GBPUSD 0.0050, USDJPY 0.50) zone ke andar (+-0.3 ATR)
    d_open       bull: entry aaj ke daily open (17:00 NY) se NEECHE (discount); bear: upar
    w_open       same, weekly open
    ema200_h1    bull: entry last closed H1 EMA200 ke upar; bear: neeche (trend ke saath)
    rsi_ext      last closed M15 RSI14: bull < 35, bear > 65 (zone mein oversold / overbought aaya)
    ote          entry impulse leg ke 0.618-0.786 retracement mein (leg = zone ka door wala wick -> fill tak ka extreme)
    corr_pair    EURUSD/GBPUSD: dusra pair bhi usi direction ke M15 zone pe +-60 min mein bhara (USDJPY = n/a)
  EXISTING (dataset mein pehle se, filter ki tarah test): has_fvg, h1_confluence, h1_flow_aligned, sweep_6h,
    sweep_24h, sweep_asia, sweep_prev_day, pd_aligned_h4, pd_aligned_d1, bias_h4_aligned, bias_d1_aligned,
    trend_tf_aligned, trend_h1_aligned.
  Tests:
    A) RAW: saare M15 touch-1 trades, confluence YES vs NO, NET 1:1 aur trail, 2016-20 / 2021-26.
    B) BOT: walk-forward bot trades (top 30% + VA + activity + exit brain, 2019-26, out-of-sample),
       sirf confluence YES wale -> PF, 2019-21 / 2022-26.
    C) ML: naye 11 confluences ML features mein daal ke walk-forward -> bot PF vs baseline.
  Pass: B mein bot se behtar DONO periods mein, PF > 1 dono mein, aur >= 20% bot trades bache. Ya C mein bot PF
    dono periods mein badhe. Data: zone_life/life5 (M5 labels). EURUSD / USDJPY / GBPUSD.
Chalana: python src/research/confluence_test.py build ; python src/research/confluence_test.py report [--ml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.features.smc import find_orderflow_zones, track_zones  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402
from src.research.zone_dataset_v2 import asian_ranges  # noqa: E402

BASE = ROOT / "data" / "research" / "zone_life" / "life5"
OUT = ROOT / "data" / "research" / "confluence"
NY = "America/New_York"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
NEAR = 0.3
NEW = ["h4_zone", "pd_level", "pw_level", "asia_level", "round_50", "d_open", "w_open", "ema200_h1", "rsi_ext",
       "ote", "corr_pair"]
OLD = ["has_fvg", "h1_confluence", "h1_flow_aligned", "sweep_6h", "sweep_24h", "sweep_asia", "sweep_prev_day",
       "pd_aligned_h4", "pd_aligned_d1", "bias_h4_aligned", "bias_d1_aligned", "trend_tf_aligned", "trend_h1_aligned"]


def asof(t, df, cols, on="close_time"):
    """Har t ke liye last row jiska `on` <= t."""
    right = df[[on] + cols].copy()
    right[on] = pd.to_datetime(right[on], utc=True).astype("datetime64[ns, UTC]")
    right = right.sort_values(on)
    left = pd.DataFrame({"t": pd.to_datetime(t, utc=True).astype("datetime64[ns, UTC]")})
    left["_i"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("t"), right, left_on="t", right_on=on, direction="backward")
    return m.sort_values("_i")[cols].to_numpy()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def level_in(level, bull, top, bot, a):
    return (level >= bot - NEAR * a) & (level <= top + NEAR * a)


def build_sym(sym):
    d = pd.read_parquet(BASE / f"{sym}_v2.parquet")
    d = d[(d.zone_tf == "M15") & (d.touch_number == 1)].reset_index(drop=True)
    m5, m15, h1, h4 = (load(sym, tf) for tf in ["M5", "M15", "H1", "H4"])
    t = d["fill_time"]
    bull = (d["direction"] == "bull").to_numpy()
    top, bot, a, entry = (d[c].to_numpy() for c in ["top", "bottom", "atr", "entry"])

    # trading day (17:00 NY roll) / week
    ny = m5.index.tz_convert(NY)
    tday = pd.Index((ny + pd.Timedelta(hours=7)).date)
    g = m5.groupby(tday)
    day = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min(), "op": g["open"].first(),
                        "start": pd.Series(m5.index, index=m5.index).groupby(tday).min()})
    day["pdh"], day["pdl"] = day["hi"].shift(1), day["lo"].shift(1)
    day = day.reset_index(drop=True)
    day["close_time"] = day["start"]
    pdh, pdl, dop = asof(t, day, ["pdh", "pdl", "op"]).T
    wk = pd.Index((ny + pd.Timedelta(hours=7)).to_period("W-SAT").astype(str))   # hafte ka roll Sunday 17:00 NY
    g = m5.groupby(wk)
    week = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min(), "op": g["open"].first(),
                         "start": pd.Series(m5.index, index=m5.index).groupby(wk).min()}).sort_values("start")
    week["pwh"], week["pwl"] = week["hi"].shift(1), week["lo"].shift(1)
    week = week.reset_index(drop=True)
    week["close_time"] = week["start"]
    pwh, pwl, wop = asof(t, week, ["pwh", "pwl", "op"]).T
    pdh, pdl, dop, pwh, pwl, wop = (np.asarray(x, float) for x in (pdh, pdl, dop, pwh, pwl, wop))

    d["pd_level"] = np.where(bull, level_in(pdl, bull, top, bot, a), level_in(pdh, bull, top, bot, a))
    d["pw_level"] = np.where(bull, level_in(pwl, bull, top, bot, a), level_in(pwh, bull, top, bot, a))
    d["d_open"] = np.where(bull, entry < dop, entry > dop)
    d["w_open"] = np.where(bull, entry < wop, entry > wop)

    # Asia range of the fill's NY date (only if fill is after 00:00 NY = range complete)
    asia = asian_ranges(m5)
    fny = t.dt.tz_convert(NY)
    key = fny.dt.date
    ah, al = key.map(asia["asia_high"]).to_numpy(float), key.map(asia["asia_low"]).to_numpy(float)
    d["asia_level"] = np.where(fny.dt.hour < 20, np.where(bull, level_in(al, bull, top, bot, a),
                                                           level_in(ah, bull, top, bot, a)), False)

    step = 0.5 if sym.endswith("JPY") else 0.005
    lvl = np.where(bull, np.ceil((bot - NEAR * a) / step) * step, np.ceil((bot - NEAR * a) / step) * step)
    d["round_50"] = lvl <= top + NEAR * a

    h1 = h1.copy()
    h1["ema200"] = h1["close"].ewm(span=200, adjust=False).mean()
    ema = asof(t, h1, ["ema200"]).ravel().astype(float)
    d["ema200_h1"] = np.where(bull, entry > ema, entry < ema)
    m15 = m15.copy()
    m15["rsi"] = rsi(m15["close"])
    r = asof(t, m15, ["rsi"]).ravel().astype(float)
    d["rsi_ext"] = np.where(bull, r < 35, r > 65)

    # OTE: leg from far wick to the extreme before fill (M5)
    hi5, lo5 = m5["high"].to_numpy(), m5["low"].to_numpy()
    idx = m5.index
    s = idx.searchsorted(d["zone_candle_time"].to_numpy())
    f = idx.searchsorted(t.to_numpy())
    ote = np.zeros(len(d), bool)
    for i in range(len(d)):
        if f[i] <= s[i]:
            continue
        if bull[i]:
            ext = hi5[s[i]:f[i]].max()
            leg = ext - bot[i]
            rt = (ext - entry[i]) / leg if leg > 0 else np.nan
        else:
            ext = lo5[s[i]:f[i]].min()
            leg = top[i] - ext
            rt = (entry[i] - ext) / leg if leg > 0 else np.nan
        ote[i] = 0.618 <= rt <= 0.786
    d["ote"] = ote

    # H4 zones (swing 5) alive at fill: created <= fill < invalidated (or 60 trading d), same direction, overlap
    z4 = track_zones(find_orderflow_zones(h4, swing_n=5, min_disp_atr=config.disp_for(sym)), h4, max_bars=360)
    zc = z4["created_at"].to_numpy(dtype="datetime64[ns]")
    zi = z4["invalidated_time"].fillna(pd.Timestamp("2100-01-01", tz="UTC")).to_numpy(dtype="datetime64[ns]")
    zend = np.minimum(zi, zc + np.timedelta64(84, "D"))
    zb = (z4["direction"] == "bull").to_numpy()
    zt, zbo = z4["top"].to_numpy(), z4["bottom"].to_numpy()
    tt = t.to_numpy(dtype="datetime64[ns]")
    h4z = np.zeros(len(d), bool)
    for i in range(len(d)):
        m = (zc <= tt[i]) & (tt[i] < zend) & (zb == bull[i]) & (zbo <= top[i]) & (zt >= bot[i])
        h4z[i] = m.any()
    d["h4_zone"] = h4z
    return d


def build():
    parts = {s: build_sym(s) for s in SYMS}
    # corr pair: EURUSD <-> GBPUSD same direction fill within 60 min
    for s, o in (("EURUSD", "GBPUSD"), ("GBPUSD", "EURUSD")):
        a, b = parts[s], parts[o]
        res = np.zeros(len(a), bool)
        for dirn in ("bull", "bear"):
            bt = np.sort(b.loc[b.direction == dirn, "fill_time"].to_numpy(dtype="datetime64[ns]"))
            ma = (a.direction == dirn).to_numpy()
            at = a.loc[ma, "fill_time"].to_numpy(dtype="datetime64[ns]")
            j = np.searchsorted(bt, at)
            near = np.zeros(len(at), bool)
            for k in (j - 1, j):
                ok = (k >= 0) & (k < len(bt))
                kk = np.clip(k, 0, len(bt) - 1)
                near |= ok & (np.abs(bt[kk] - at) <= np.timedelta64(60, "m"))
            res[ma] = near
        a["corr_pair"] = res
    parts["USDJPY"]["corr_pair"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    for s, d in parts.items():
        d.to_parquet(OUT / f"{s}_conf.parquet")
        print(s, len(d), {c: round(float(d[c].mean()) * 100, 1) for c in NEW})


def pf(r):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def report(ml=False):
    from src.ml import walkforward as wf
    conf = pd.concat([pd.read_parquet(OUT / f"{s}_conf.parquet") for s in SYMS], ignore_index=True)
    key = ["symbol", "zone_id", "touch_number"]
    conf = conf[key + NEW]
    d = wf.load(SYMS, folder=BASE)
    d = d.merge(conf, on=key, how="left")
    for c in NEW:
        d[c] = d[c].astype("boolean").fillna(False).astype(bool)
    for c in wf.CATEGORICAL:
        d[c] = d[c].astype("category")
    for c in OLD:
        d["is_" + c] = d[c].fillna(0).astype(float) > 0
    for c in NEW:
        d["is_" + c] = d[c]
    d["period"] = np.where(d.fill_time.dt.year <= 2020, "16-20", "21-26")

    raw = d[(d.zone_tf == "M15") & (d.touch_number == 1)]
    rows = []
    for c in NEW + OLD:
        row = dict(conf=c, kind="new" if c in NEW else "old", yes_pct=round(100 * raw["is_" + c].mean(), 1))
        for per, g in raw.groupby("period"):
            row[f"yes_tp1_{per}"], row[f"no_tp1_{per}"] = pf(g.loc[g["is_" + c], "r_tp1"]), pf(g.loc[~g["is_" + c], "r_tp1"])
            row[f"yes_trail_{per}"] = pf(g.loc[g["is_" + c], "r_trail"])
        rows.append(row)
    A = pd.DataFrame(rows)

    w = wf.walk_forward(d, 2019, log=lambda *a: None)
    bot = w[(w.zone_tf == "M15") & (w.touch_number == 1) & w["keep_0.3"] & w.va_ok & w.act_ok]
    p1, p2 = bot.test_year <= 2021, bot.test_year >= 2022
    rows = [dict(conf="BOT (all)", trades=len(bot), keep_pct=100.0, pf=pf(bot.r_ml), pf_19_21=pf(bot.loc[p1, "r_ml"]),
                 pf_22_26=pf(bot.loc[p2, "r_ml"]), total_r=round(bot.r_ml.sum(), 1))]
    for c in NEW + OLD:
        for val in (True, False):
            g = bot[bot["is_" + c] == val]
            rows.append(dict(conf=f"{c}={'YES' if val else 'NO'}", trades=len(g), keep_pct=round(100 * len(g) / len(bot), 1),
                             pf=pf(g.r_ml), pf_19_21=pf(g.loc[g.test_year <= 2021, "r_ml"]),
                             pf_22_26=pf(g.loc[g.test_year >= 2022, "r_ml"]), total_r=round(g.r_ml.sum(), 1)))
    B = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 200)
    print("== A) RAW M15 touch 1 (NET PF, YES vs NO) ==\n", A.to_string(index=False))
    print("\n== B) BOT trades (out-of-sample) filtered by confluence ==\n", B.to_string(index=False))
    A.to_csv(OUT / "report_raw.csv", index=False)
    B.to_csv(OUT / "report_bot.csv", index=False)

    if ml:
        base_f = list(wf.FEATURES)
        wf.FEATURES[:] = base_f + NEW
        for c in NEW:
            d[c] = d[c].astype(float)
        try:
            w2 = wf.walk_forward(d, 2019, log=lambda *a: None)
        finally:
            wf.FEATURES[:] = base_f
        b2 = w2[(w2.zone_tf == "M15") & (w2.touch_number == 1) & w2["keep_0.3"] & w2.va_ok & w2.act_ok]
        C = pd.DataFrame([dict(model=m, trades=len(x), pf=pf(x.r_ml), pf_19_21=pf(x.loc[x.test_year <= 2021, "r_ml"]),
                               pf_22_26=pf(x.loc[x.test_year >= 2022, "r_ml"]), total_r=round(x.r_ml.sum(), 1))
                          for m, x in (("ML now", bot), ("ML + 11 confluences", b2))])
        print("\n== C) ML with confluence features ==\n", C.to_string(index=False))
        C.to_csv(OUT / "report_ml.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    r = sp.add_parser("report")
    r.add_argument("--ml", action="store_true")
    a = ap.parse_args()
    build() if a.cmd == "build" else report(a.ml)
