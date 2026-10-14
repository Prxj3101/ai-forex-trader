r"""
GOLD FIB 7 - ML (15 Oct 2026). He asked: train the ML on the best setups so far, train/test properly, report the best
PF for 15m, H1 and both together.
SETUPS   M15 K2.5 zones (round-3 pool) | H1 K2.5 | H1 K3.5 (best H1 rules). First-tap fill (FT2), clean filters
         (fib 0 not broken, no M5 pre-tap), 10 days valid. 2016-20 = v3 parquet rows; 2021-26 run here (same code).
EXIT     SL 1.0, TP 1:1 (his way) = what the model is trained and judged on. Costs $7 RT + $0.05 slip + Dukascopy spread.
FEATURES known at the fill moment only (no look-ahead): move size/ATR, size % of price, leg bars, leg efficiency,
         counter-candle share, max pullback inside the leg, BOS / session / sweep, side, H4 trend distance, daily ATR
         rank, hours from zone creation to fill, bars from move end to fill, NY hour + weekday of the fill, approach
         speed into the zone, + for M15 the "his eye" score (model trained on his 200 M15 YES/NO labels).
MODELS   (1) OUTCOME model: gradient boosting predicts win (r > 0).  (2) EYE model: predicts his YES (M15 only).
TRAIN    2016-20 only, walk-forward: fit 2016-17 -> score 2018, 2016-18 -> 2019, 2016-19 -> 2020 (out-of-sample scores).
         Threshold = take the top q of scores; q chosen on these 2018-20 scores (needs >= 20 trades/yr).
         Then refit on 2016-20, FREEZE, score 2021-26 ONCE.
PASS     2021-26 PF >= 1.10.
Chalana: python src\research\gold_fib_ml.py trades | features | fit
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import gold_fib as gf  # noqa: E402
from src.research import gold_fib_v3 as v3  # noqa: E402
from src.research.gold_fib_big import clean_flag, m5_arrays  # noqa: E402

OUT = ROOT / "data/research/gold_fib_ml"
SETS = [("M15", 2.5), ("H1", 2.5), ("H1", 3.5)]
NY = "America/New_York"


def trades():
    OUT.mkdir(parents=True, exist_ok=True)
    hi, lo, op, ct = m5_arrays()
    m5 = None
    for tf, k in SETS:
        tag = f"{tf}_K{k}"
        old = OUT / f"{tag}_2016_2020.parquet"
        if not old.exists():
            d = pd.read_parquet(ROOT / f"data/research/gold_fib_v3/K{k}_{tf}.parquet")
            d = d[d.fill_FT2.notna()].copy()
            d["clean"] = [clean_flag(r, hi, lo, op, ct) for r in d.to_dict("records")]
            d.to_parquet(old)
            print(tag, "2016-20 rows", len(d), flush=True)
        zf = OUT / f"{tag}_zones_all.parquet"
        if not zf.exists():
            if m5 is None:
                m5 = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"])
                sess, h4 = gf.session_table(m5), gf.bars("H4")
            v3.zones_k(k, tf, gf.bars(tf), sess, h4).to_parquet(zf)
            print(tag, "zones saved", flush=True)
            return  # keep each call short
        z = pd.read_parquet(zf)
        for y in range(2021, 2027):
            f = OUT / f"{tag}_{y}.parquet"
            if f.exists():
                continue
            t0 = time.time()
            rows = []
            for r in z[z.created.dt.year == y].to_dict("records"):
                res = {**r, **v3.trade(pd.Series(r))}
                if res.get("fill_FT2") is None or pd.isna(res.get("fill_FT2")):
                    continue
                res["clean"] = clean_flag(res, hi, lo, op, ct)
                rows.append(res)
            pd.DataFrame(rows).to_parquet(f)
            print(tag, y, len(rows), f"{time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > 60:
                return
    print("all trades done")


def _ms(idx):
    return pd.DatetimeIndex(idx).tz_convert(None).to_numpy().astype("datetime64[ms]").astype("int64")


def features():
    h4 = gf.bars("H4")
    h4["ema"] = h4.close.ewm(span=50, adjust=False).mean()
    h4ct = _ms(h4.close_time)
    d1 = gf.bars("D1")
    d1["rank"] = d1.atr.rolling(250, min_periods=50).rank(pct=True)
    d1ct = _ms(d1.close_time)
    for tf, k in SETS:
        tag = f"{tf}_K{k}"
        d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob(f"{tag}_20*.parquet"))], ignore_index=True)
        d = d[d.clean & d.r_FT2_C_1R.notna()].copy().reset_index(drop=True)
        b = gf.bars(tf)
        bo, bh, bl, bc = b.open.to_numpy(), b.high.to_numpy(), b.low.to_numpy(), b.close.to_numpy()
        bop, bct = _ms(b.index), _ms(b.close_time)
        F = []
        for r in d.to_dict("records"):
            bull, fill = r["bull"], int(r["fill_FT2"])
            i1 = np.searchsorted(bop, int(r["t_start"].timestamp() * 1000))
            i2 = np.searchsorted(bop, int(r["t_end"].timestamp() * 1000))
            seg = slice(i1, i2 + 1)
            size = r["H"] - r["L"]
            rng = (bh[seg] - bl[seg]).sum()
            sg = 1 if bull else -1
            against = float(((bc[seg] - bo[seg]) * sg < 0).mean())
            if bull:
                pull = float((np.maximum.accumulate(bh[seg]) - bl[seg]).max() / size)
            else:
                pull = float((bh[seg] - np.minimum.accumulate(bl[seg])).max() / size)
            j = np.searchsorted(bct, fill, side="right") - 1          # last closed bar before the fill
            app = (bc[j] - bc[max(j - 4, 0)]) * -sg / max(r["atr"], 1e-9)  # speed of the pullback toward the zone
            k4 = np.searchsorted(h4ct, fill, side="right") - 1
            h4d = (h4.close.iloc[k4] - h4.ema.iloc[k4]) / h4.atr.iloc[k4] * sg
            kd = np.searchsorted(d1ct, fill, side="right") - 1
            ft = pd.Timestamp(fill, unit="ms", tz="UTC").tz_convert(NY)
            F.append(dict(satr=size / r["atr"], pct=100 * size / r["entry"], leg_bars=i2 - i1 + 1, eff=size / max(rng, 1e-9),
                          against=against, max_pull=pull, bos=int(r["bos"]), session=int(r["session"]), sweep=int(r["sweep"]),
                          bull=int(bull), h4_dist=h4d, d1_rank=d1["rank"].iloc[kd], hrs_to_fill=(fill / 1000 - r["created"].timestamp()) / 3600,
                          bars_to_fill=j - i2, approach=app, ny_hour=ft.hour, weekday=ft.weekday(),
                          start_hour=r["t_start"].tz_convert(NY).hour))
        out = pd.concat([d, pd.DataFrame(F).add_prefix("f_")], axis=1)
        out["year"] = pd.to_datetime(out.fill_FT2, unit="ms").dt.year
        out.to_parquet(OUT / f"feat_{tag}.parquet")
        print(tag, len(out), out.groupby(out.year <= 2020).size().to_dict(), flush=True)


def _hgb(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=25,
                                          l2_regularization=1.0, random_state=seed)


def pf(s):
    s = pd.Series(s).dropna()
    return round(float(s[s > 0].sum() / -s[s < 0].sum()), 2) if (s < 0).any() else np.nan


def eye_scores(m):
    """His-eye model: trained on his 200 M15 YES/NO labels (2016-20). Labelled rows get 5-fold out-of-fold scores."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    lab = json.loads((OUT / "labels_prajwal_m15.json").read_text())
    key = pd.concat([pd.read_parquet(ROOT / "data/research/gold_fib/review3/key.parquet"),
                     pd.read_parquet(ROOT / "data/research/gold_fib/review3/key_extra.parquet")])
    key = key[key.tf == "M15"]
    yes = {i: 1 for i in lab["yes"].split(",")} | {i: 0 for i in lab["no"].split(",")}
    key = key[key.id.isin(yes)]
    km = {(int(r.created.value), bool(r.bull)): yes[r.id] for r in key.itertuples()}
    m["eye_y"] = [km.get((int(c.value), bool(b)), np.nan) for c, b in zip(m.created, m.bull)]
    F = [c for c in m.columns if c.startswith("f_")]
    L = m[m.eye_y.notna()]
    print(f"eye labels matched: {len(L)} / {len(key)}  (YES {int(L.eye_y.sum())})")
    oof = pd.Series(np.nan, index=L.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(L, L.eye_y):
        mod = _hgb().fit(L[F].iloc[tr], L.eye_y.iloc[tr])
        oof.iloc[te] = mod.predict_proba(L[F].iloc[te])[:, 1]
    print(f"eye model: out-of-fold AUC vs his answers = {roc_auc_score(L.eye_y, oof):.2f}  (0.5 = guessing)")
    full = _hgb().fit(L[F], L.eye_y)
    sc = pd.Series(full.predict_proba(m[F])[:, 1], index=m.index)
    sc.loc[oof.index] = oof
    return sc


def pick(scores, years, q):
    """Take the top q of scores, threshold for year Y = (1-q) quantile of the scores of year Y-1 (no future info)."""
    take = pd.Series(False, index=scores.index)
    for y in sorted(years.unique()):
        prev = scores[years == y - 1]
        if len(prev) < 20:
            continue
        thr = prev.quantile(1 - q)
        take |= (years == y) & (scores >= thr)
    return take


def fit():
    QS = [0.2, 0.3, 0.4, 0.5]
    res, picks = [], {}
    for tf, k in SETS:
        tag = f"{tf}_K{k}"
        m = pd.read_parquet(OUT / f"feat_{tag}.parquet")
        m["win"] = (m.r_FT2_C_1R > 0).astype(int)
        R = m.r_FT2_C_1R
        tr_mask, ho_mask = m.year <= 2020, m.year >= 2021
        if tf == "M15":
            m["f_eye"] = eye_scores(m)
        F = [c for c in m.columns if c.startswith("f_")]
        yrs_in, yrs_ho = 3, m[ho_mask].year.nunique() - 0.25
        res.append(dict(set=tag, strategy="rules only", q="-", in_pf_18_20=pf(R[m.year.between(2018, 2020)]),
                        in_tr_yr=round(m.year.between(2018, 2020).sum() / 3), hold_pf_21_26=pf(R[ho_mask]),
                        hold_tr_yr=round(ho_mask.sum() / yrs_ho), hold_R=round(R[ho_mask].sum(), 1)))
        picks[(tag, "rules only")] = m[["year"]].assign(R=R, take=True)
        if tf == "M15":   # his eye: q fixed a priori = his YES rate (77/200)
            t = pick(m.f_eye, m.year, 0.385)
            res.append(dict(set=tag, strategy="his-eye model", q=0.385, in_pf_18_20=pf(R[t & m.year.between(2018, 2020)]),
                            in_tr_yr=round((t & m.year.between(2018, 2020)).sum() / 3), hold_pf_21_26=pf(R[t & ho_mask]),
                            hold_tr_yr=round((t & ho_mask).sum() / yrs_ho), hold_R=round(R[t & ho_mask].sum(), 1)))
            picks[(tag, "his-eye model")] = m[["year"]].assign(R=R, take=t)
        # outcome model, walk-forward inside 2016-20
        oof = pd.Series(np.nan, index=m.index)
        for ty in [2018, 2019, 2020]:
            trn = m.year < ty
            mod = _hgb().fit(m.loc[trn, F], m.loc[trn, "win"])
            oof[m.year == ty] = mod.predict_proba(m.loc[m.year == ty, F])[:, 1]
        best = None
        for q in QS:
            t = pick(oof.fillna(-1), m.year, q) & m.year.between(2019, 2020)  # 2018 has no prev-year OOF scores
            n_yr = t.sum() / 2
            p = pf(R[t])
            if n_yr >= 20 and (best is None or p > best[1]):
                best = (q, p, n_yr)
        if best is None:   # small set (H1 K3.5): no q gives >= 20/yr -> use the widest q
            t = pick(oof.fillna(-1), m.year, QS[-1]) & m.year.between(2019, 2020)
            best = (QS[-1], pf(R[t]), t.sum() / 2)
        q = best[0]
        final = _hgb().fit(m.loc[tr_mask, F], m.loc[tr_mask, "win"])
        sc = oof.copy()
        sc[ho_mask] = final.predict_proba(m.loc[ho_mask, F])[:, 1]
        t = pick(sc.fillna(-1), m.year, q)
        res.append(dict(set=tag, strategy="outcome model", q=q, in_pf_18_20=best[1], in_tr_yr=round(best[2]),
                        hold_pf_21_26=pf(R[t & ho_mask]), hold_tr_yr=round((t & ho_mask).sum() / yrs_ho), hold_R=round(R[t & ho_mask].sum(), 1)))
        picks[(tag, "outcome model")] = m[["year"]].assign(R=R, take=t)
        imp = pd.Series(np.abs(np.corrcoef(np.c_[m.loc[tr_mask, F].fillna(0).to_numpy(), m.loc[tr_mask, "win"]].T)[-1, :-1]), index=F)
        print(tag, "feature link to win (2016-20, |corr|):", imp.sort_values(ascending=False).head(5).round(3).to_dict(), flush=True)
    r = pd.DataFrame(res)
    pd.set_option("display.width", 220)
    print(r.to_string(index=False))
    print("\nCOMBINED 15m + H1, 2021-26 holdout:")
    for a, b in [(("M15_K2.5", "outcome model"), ("H1_K2.5", "rules only")), (("M15_K2.5", "his-eye model"), ("H1_K2.5", "rules only")), (("M15_K2.5", "rules only"), ("H1_K3.5", "rules only")), (("M15_K2.5", "his-eye model"), ("H1_K3.5", "rules only")),
                 (("M15_K2.5", "outcome model"), ("H1_K3.5", "outcome model")), (("M15_K2.5", "his-eye model"), ("H1_K3.5", "outcome model"))]:
        x = pd.concat([picks[a], picks[b]])
        h = x[(x.year >= 2021) & x["take"]]
        i = x[x.year.between(2019, 2020) & x["take"]]
        print(f"  15m {a[1]:14s} + {b[0]} {b[1]:14s}: 2019-20 PF {pf(i.R)} ({round(len(i)/2)}/yr) | 2021-26 PF {pf(h.R)} ({round(len(h)/5.75)}/yr, {round(h.R.sum(),1)}R)")
        print("     per year:", h.groupby("year").R.sum().round(1).to_dict())


if __name__ == "__main__":
    {"trades": trades, "features": features, "fit": fit}[sys.argv[1]]()
