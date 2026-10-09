r"""
ORB + ML TEST (9 Oct 2026). Rules fixed BEFORE running.
Trades: tick ORB breakouts from orb_nr_test (ASIA / LON30 / LON60, EUR/GBP/JPY, SL other side), exits 1R / 2R / EOD.
Features (all known at the break tick, no lookahead):
  compression: nr3..nr7, inside, w_ratio (range / avg prev 20), range_atr (range / D1 ATR14)
  trend: h4_aligned (last H4 BOS same dir), d1_mom_aligned (sign of 20-day change), prevday_aligned (yesterday candle dir)
  location: prevday_pos (entry inside yesterday's range, 0 = against side, 1 = with side), asia_pos (London rows: entry vs Asia range)
  time: break_delay_min (minutes after range end), weekday
  activity: act_15m (M5 tick volume last 15 min / 24h average), spread_r (spread / risk), risk_pips
  plus symbol, range (categorical)
MODEL: LightGBM classifier (net R > 0), one model for all pairs/ranges, per exit. Walk-forward: train on all years < Y,
  test year Y, Y = 2019..2026. Keep trades with score >= train-score quantile (keep 50% / 30%). 3 seeds.
COMPARE with simple rules on the same years: no filter, NR4, NR7, TIGHT.
PASS (2021-26): net PF >= 1.10, PF > 1 in 2021-23 and 2024-26, >= 5/6 years positive, >= 2/3 pairs > 1, >= 150 trades/yr,
  AND better than the best simple rule by > 0.03 PF (seed noise), on the average of 3 seeds.
Chalana: python src\research\orb_ml_test.py features | python src\research\orb_ml_test.py wf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.smc import atr, structure_breaks  # noqa: E402
from src.research import orb_nr_test as nr  # noqa: E402

FEAT = ROOT / "data/research/orb_ml_features.parquet"
FEATURES = ["nr3", "nr4", "nr5", "nr6", "nr7", "inside", "w_ratio", "range_atr", "h4_aligned", "d1_mom_aligned",
            "prevday_aligned", "prevday_pos", "asia_pos", "break_delay_min", "weekday", "act_15m", "spread_r", "risk_pips",
            "symbol", "range"]
EXITS = ("1R", "2R", "EOD")
pf = nr.pf


def features():
    d = nr.load()
    d = d[(d.broke == 1) & d.BRK_risk.notna()].copy()
    asia = d[d.range == "ASIA"].set_index(["symbol", "day"])[["hi", "lo"]].rename(columns={"hi": "a_hi", "lo": "a_lo"})
    d = d.join(asia, on=["symbol", "day"])
    out = []
    for sym, g in d.groupby("symbol"):
        g = g.copy()
        bull = (g.direction == "bull").to_numpy(); sg = np.where(bull, 1, -1)
        bt = pd.to_datetime(g.break_ms, unit="ms", utc=True)
        rt = pd.to_datetime(g.range_end_ms, unit="ms", utc=True)
        d1 = pd.read_parquet(ROOT / f"data/processed/{sym}_D1.parquet"); d1["atr"] = atr(d1)
        ct = pd.DatetimeIndex(d1.close_time).values
        di = np.searchsorted(ct, rt.values.astype("datetime64[ns]"), side="right") - 1
        rng = (d1.high - d1.low).to_numpy(); H, L, O, C = (d1[c].to_numpy() for c in ("high", "low", "open", "close"))
        for n in range(3, 8):
            g[f"nr{n}"] = [int(rng[i] <= rng[i - n + 1:i + 1].min()) if i >= 8 else np.nan for i in di]
        g["range_atr"] = g.w.to_numpy() / d1.atr.to_numpy()[di]
        mom = np.sign(C[di] - C[np.maximum(di - 20, 0)])
        g["d1_mom_aligned"] = (mom == sg).astype(int)
        g["prevday_aligned"] = (np.sign(C[di] - O[di]) == sg).astype(int)
        pos = (g.entry.to_numpy() - L[di]) / np.maximum(H[di] - L[di], 1e-12)
        g["prevday_pos"] = np.where(bull, pos, 1 - pos)
        ap = (g.entry - g.a_lo) / (g.a_hi - g.a_lo)
        g["asia_pos"] = np.where(g.range == "ASIA", np.nan, np.where(bull, ap, 1 - ap))
        h4 = pd.read_parquet(ROOT / f"data/processed/{sym}_H4.parquet")
        br = structure_breaks(h4, 2); brt = pd.DatetimeIndex(br["time"]); brd = br["direction"].astype(str).to_numpy()
        k = brt.searchsorted(bt, side="right") - 1
        g["h4_aligned"] = [int((brd[i] == "bull") == b) if i >= 0 else np.nan for i, b in zip(k, bull)]
        g["break_delay_min"] = (g.break_ms - g.range_end_ms) / 60000
        g["weekday"] = pd.to_datetime(g.day).dt.weekday
        m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["volume", "spread"])
        v = m5.volume.to_numpy(); cs = np.concatenate([[0], np.cumsum(v)])
        j = np.searchsorted(m5.index.values, bt.values.astype("datetime64[ns]"), side="right") - 1  # bar containing break
        j = np.clip(j, 289, len(v) - 1)
        last3 = (cs[j] - cs[j - 3]) / 3; day288 = (cs[j] - cs[j - 288]) / 288
        g["act_15m"] = last3 / np.maximum(day288, 1e-9)
        g["spread_r"] = m5.spread.to_numpy()[j] / g.BRK_risk.to_numpy()
        g["risk_pips"] = g.BRK_risk / nr.PIP[sym]
        out.append(g)
    d = pd.concat(out)
    keep = ["symbol", "day", "range", "direction", "when", "break_ms"] + [f for f in FEATURES if f not in ("symbol", "range")] + \
           [f"BRK_{e}_net" for e in EXITS]
    d[keep].to_parquet(FEAT)
    print("features", len(d), d[FEATURES[:-2]].isna().mean().round(3).to_dict())


def wf(seeds=(1, 2, 3)):
    import lightgbm as lgb
    d = pd.read_parquet(FEAT)
    d["year"] = d.when.dt.year
    X = d[FEATURES].copy()
    X["symbol"] = pd.Categorical(X.symbol, categories=["EURUSD", "GBPUSD", "USDJPY"])
    X["range"] = pd.Categorical(X["range"], categories=["ASIA", "LON30", "LON60"])
    res = {}
    for ex in EXITS:
        y = d[f"BRK_{ex}_net"]
        for seed in seeds:
            sc = pd.Series(np.nan, index=d.index); th = {}
            for Y in range(2019, 2027):
                tr, te_ = (d.year < Y) & y.notna(), d.year == Y
                m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
                m.fit(X[tr], (y[tr] > 0).astype(int))
                s_tr = m.predict_proba(X[tr])[:, 1]
                th[Y] = {k: np.quantile(s_tr, 1 - k) for k in (0.5, 0.3)}
                sc[te_] = m.predict_proba(X[te_])[:, 1]
            for k in (0.5, 0.3):
                keep = pd.Series(False, index=d.index)
                for Y in range(2019, 2027):
                    keep |= (d.year == Y) & (sc >= th[Y][k])
                res[(ex, seed, k)] = keep
            if ex == "EOD" and seed == 1:
                imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
                print("feature importance (EOD, last model):", (imp / imp.sum()).round(3).head(12).to_dict())
    return d, res


def judge(x, col):
    x = x[x[col].notna()]
    y = x.year; yrs = x.groupby(y)[col].sum()
    pairs = [pf(q[col]) for _, q in x.groupby("symbol")]
    h1, h2 = pf(x.loc[y <= 2023, col]), pf(x.loc[y >= 2024, col])
    return dict(n_yr=round(len(x) / 5.75), pf=pf(x[col]), pf_21_23=h1, pf_24_26=h2,
                pos_yrs=f"{int((yrs > 0).sum())}/{yrs.size}", pairs=pairs)


def report():
    d, res = wf()
    T = d.year >= 2021
    learn = d[d.year <= 2020]
    q30 = {r: nr.load().pipe(lambda z: z[(z.when.dt.year <= 2020) & (z.range == r)].w_ratio.quantile(0.3)) for r in ["ASIA", "LON30", "LON60"]}
    tight = d.apply(lambda r: r.w_ratio <= q30[r["range"]], axis=1)
    rows = []
    for ex in EXITS:
        col = f"BRK_{ex}_net"
        for rg in ["ALL", "ASIA", "LON30", "LON60"]:
            R = (d["range"] == rg) if rg != "ALL" else pd.Series(True, index=d.index)
            for name, m in (("rule: none", R), ("rule: NR4", R & (d.nr4 == 1)), ("rule: NR7", R & (d.nr7 == 1)),
                            ("rule: TIGHT", R & tight)):
                rows.append(dict(exit=ex, range=rg, filter=name, **judge(d[m & T], col)))
            for k in (0.5, 0.3):
                pfs = [judge(d[res[(ex, s, k)] & R & T], col) for s in (1, 2, 3)]
                avg = {kk: pfs[0][kk] for kk in ("n_yr", "pos_yrs", "pairs")}
                rows.append(dict(exit=ex, range=rg, filter=f"ML keep {int(k*100)}% (3-seed avg)", **avg,
                                 pf=round(np.mean([p["pf"] for p in pfs]), 3), pf_21_23=round(np.mean([p["pf_21_23"] for p in pfs]), 3),
                                 pf_24_26=round(np.mean([p["pf_24_26"] for p in pfs]), 3), seeds=[p["pf"] for p in pfs]))
                # ML on top of compression
                pfs = [judge(d[res[(ex, s, k)] & R & T & ((d.nr4 == 1) | tight)], col) for s in (1, 2, 3)]
                rows.append(dict(exit=ex, range=rg, filter=f"ML {int(k*100)}% + (NR4 or TIGHT)", n_yr=pfs[0]["n_yr"],
                                 pos_yrs=pfs[0]["pos_yrs"], pairs=pfs[0]["pairs"], pf=round(np.mean([p["pf"] for p in pfs]), 3),
                                 pf_21_23=round(np.mean([p["pf_21_23"] for p in pfs]), 3),
                                 pf_24_26=round(np.mean([p["pf_24_26"] for p in pfs]), 3), seeds=[p["pf"] for p in pfs]))
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "data/research/orb_ml.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    if sys.argv[1] == "features":
        features()
    else:
        print(report().to_string(index=False))
