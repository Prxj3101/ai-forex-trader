r"""
SESSION VOLUME PROFILE TEST (plan agreed 9 Oct 2026, rules fixed before running). Tick-labelled v2, net of costs.
Features (only FINISHED profiles before the zone exists; current session open known at fill):
  A vp_sess_edge   : previous session VAL (BUY) / VAH (SELL) within the zone +-1 ATR
  B vp_sess_poc    : previous session POC within the zone +-1 ATR
  C vp_day_edge    : previous DAY VAL (BUY) / VAH (SELL) within the zone +-1 ATR
    vp_day_pos     : zone position in previous day value area (0 = cheap edge for the trade, 1 = expensive edge)
  D vp_density     : previous-day volume at the zone price / median volume of that profile (LVN < 1 < HVN)
  E vp_naked_r     : nearest naked (untouched) daily POC of the last 10 days IN the trade direction, distance in R
    vp_naked_ok    : such a naked POC between 0.5R and 5R
  F vp_open_side   : current session opened outside previous session VA in the trade direction (+1), against (-1), inside (0)
Rules: thresholds / side chosen on 2016-2020, checked on 2021-2026.
PASS: 2021-26 net PF >= 1.10, >= 5 of 6 years positive, >= 2 of 3 pairs > 1, >= 150 trades.
Also: combo H1 + H4 trend + 1:2 with each filter, and ML walk-forward with these features (3 seeds, top 20%).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402
from src.features import session_vp as svp  # noqa: E402

pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 2) if (r < 0).any() else np.nan  # noqa: E731
VP = ["vp_sess_edge", "vp_sess_poc", "vp_day_edge", "vp_day_pos", "vp_density", "vp_naked_r", "vp_naked_ok", "vp_open_side"]


def features(d):
    out = []
    for sym, g in d.groupby(d.symbol.astype(str)):
        prof = svp.load(sym)
        pip = svp.PIP[sym]
        S = prof[prof.kind != "DAY"].sort_values("start_ms").reset_index(drop=True)
        D = prof[prof.kind == "DAY"].sort_values("start_ms").reset_index(drop=True)
        m5 = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["high", "low", "open"])
        t5 = m5.index.as_unit("ns").asi8 // 1_000_000
        h5, l5 = m5.high.to_numpy(), m5.low.to_numpy()
        g = g.copy()
        c_ms = g.created_at.astype("int64").to_numpy() // 1_000_000 if g.created_at.dtype.kind == "M" else None
        c_ms = (pd.DatetimeIndex(g.created_at).as_unit("ns").asi8 // 1_000_000)
        f_ms = (pd.DatetimeIndex(g.fill_time).as_unit("ns").asi8 // 1_000_000)
        si = np.searchsorted(S.start_ms.values, c_ms, side="right") - 2      # last FINISHED session
        di = np.searchsorted(D.start_ms.values, c_ms, side="right") - 2      # last FINISHED day
        fi = np.searchsorted(S.start_ms.values, f_ms, side="right") - 1      # session that contains the fill
        bull = (g.direction.astype(str) == "bull").to_numpy()
        lo_b = (g.bottom - g.atr).to_numpy(); hi_b = (g.top + g.atr).to_numpy()
        inb = lambda v: (v >= lo_b) & (v <= hi_b)  # noqa: E731
        sp, dp = S.iloc[np.clip(si, 0, None)], D.iloc[np.clip(di, 0, None)]
        g["vp_sess_edge"] = np.where(bull, inb(sp.val.values), inb(sp.vah.values)).astype(float)
        g["vp_sess_poc"] = inb(sp.poc.values).astype(float)
        g["vp_day_edge"] = np.where(bull, inb(dp.val.values), inb(dp.vah.values)).astype(float)
        ref = np.where(bull, g.bottom, g.top)
        width = np.maximum(dp.vah.values - dp.val.values, pip)
        g["vp_day_pos"] = np.where(bull, (ref - dp.val.values) / width, (dp.vah.values - ref) / width)
        dens = []
        for (b0, cnt), lo, hi in zip(zip(dp.bin0.values, dp.counts.values), g.bottom.values, g.top.values):
            a, b = int(round(lo / pip)) - b0, int(round(hi / pip)) - b0
            seg = cnt[max(a, 0):max(min(b + 1, len(cnt)), 0)]
            med = np.median(cnt[cnt > 0]) if (cnt > 0).any() else np.nan
            dens.append(float(seg.mean() / med) if len(seg) and med else np.nan)
        g["vp_density"] = dens
        # E: naked daily POCs (last 10 finished days), untouched from their day end until the fill
        nk = []
        for k in range(len(g)):
            best = np.nan
            if di[k] >= 1:
                e = g.entry.values[k]; risk = g.risk.values[k]; fm = f_ms[k]
                for j in range(max(di[k] - 9, 0), di[k] + 1):
                    p, end = D.poc.values[j], D.end_ms.values[j]
                    a, b = np.searchsorted(t5, end), np.searchsorted(t5, fm)
                    if b > a and (l5[a:b].min() <= p <= h5[a:b].max()):
                        continue                                   # touched -> not naked
                    dist = ((p - e) if bull[k] else (e - p)) / risk
                    if dist > 0 and (np.isnan(best) or dist < best):
                        best = dist
            nk.append(best)
        g["vp_naked_r"] = nk
        g["vp_naked_ok"] = ((g.vp_naked_r >= 0.5) & (g.vp_naked_r <= 5)).astype(float)
        # F: current session open vs previous session value area
        cur = S.iloc[np.clip(fi, 0, None)]; prv = S.iloc[np.clip(fi - 1, 0, None)]
        o_idx = np.clip(np.searchsorted(t5, cur.start_ms.values) , 0, len(t5) - 1)
        opn = m5.open.to_numpy()[o_idx]
        side = np.where(opn > prv.vah.values, 1, np.where(opn < prv.val.values, -1, 0))
        g["vp_open_side"] = np.where(bull, side, -side).astype(float)
        out.append(g)
    return pd.concat(out)


def judge(x, col="r"):
    y = x.fill_time.dt.year
    yrs = x.groupby(y)[col].sum()
    pairs = [pf(q[col]) for _, q in x.groupby(x.symbol.astype(str))]
    ok = len(x) >= 150 and pf(x[col]) >= 1.10 and (yrs > 0).sum() >= 5 and sum(p > 1 for p in pairs) >= 2
    return dict(n=len(x), pf=pf(x[col]), pos_years=f"{int((yrs > 0).sum())}/{yrs.size}", pairs=pairs, PASS=ok)


def rules(d):
    rows = []
    groups = {"M15 touch1": (d.zone_tf.astype(str) == "M15") & (d.touch_number == 1),
              "H1 touch1": (d.zone_tf.astype(str) == "H1") & (d.touch_number == 1),
              "H1 + H4 trend": (d.zone_tf.astype(str) == "H1") & (d.touch_number == 1) & (d.bias_h4_aligned == 1)}
    for gname, gm in groups.items():
        g = d[gm]
        learn, test = g[g.fill_time.dt.year <= 2020], g[g.fill_time.dt.year >= 2021]
        for ex in ("r_tp1", "r_tp2", "r_trail"):
            rows.append(dict(group=gname, exit=ex, filter="(none)", learn_pf=pf(learn[ex]), **judge(test.assign(r=test[ex]))))
            for f in VP:
                lv = learn[f].dropna()
                if lv.nunique() <= 3:                               # flag / category: pick the better value on 2016-20
                    vals = [v for v in lv.unique() if (learn[f] == v).sum() >= 100]
                    if not vals:
                        continue
                    v = max(vals, key=lambda v: pf(learn.loc[learn[f] == v, ex]))
                    m_l, m_t, lab = learn[f] == v, test[f] == v, f"{f} = {v:g}"
                else:
                    lo, hi = lv.quantile(0.4), lv.quantile(0.6)
                    use_lo = pf(learn.loc[learn[f] <= lo, ex]) >= pf(learn.loc[learn[f] >= hi, ex])
                    m_l = (learn[f] <= lo) if use_lo else (learn[f] >= hi)
                    m_t = (test[f] <= lo) if use_lo else (test[f] >= hi)
                    lab = f"{f} {'<=' if use_lo else '>='} {(lo if use_lo else hi):.2f}"
                x = test[m_t]
                rows.append(dict(group=gname, exit=ex, filter=lab, learn_pf=pf(learn.loc[m_l, ex]), **judge(x.assign(r=x[ex]))))
    return pd.DataFrame(rows)


def ml(d):
    import lightgbm as lgb
    feats = wf.FEATURES + VP
    rows = []
    for tf in ("M15", "H1"):
        g = d[(d.zone_tf.astype(str) == tf) & (d.touch_number == 1)].sort_values("fill_time")
        for ex in ("r_tp1", "r_tp2", "r_trail"):
            parts = []
            for seed in (1, 2, 3):
                for year in range(2019, 2027):
                    st = pd.Timestamp(f"{year}-01-01", tz="UTC")
                    tr, te_ = g[g.fill_time < st - pd.Timedelta(days=5)], g[g.fill_time.dt.year == year]
                    m = lgb.LGBMClassifier(**dict(wf.PARAMS, random_state=seed)).fit(tr[feats], (tr[ex] > 0).astype(int))
                    thr = np.quantile(m.predict_proba(tr[feats])[:, 1], 0.8)
                    sel = te_[m.predict_proba(te_[feats])[:, 1] >= thr]
                    parts.append(sel.assign(r=sel[ex]))
            x = pd.concat(parts)
            j = judge(x); j["n"] //= 3
            rows.append(dict(group=f"ML {tf} touch1 top20% (+VP)", exit=ex, filter="walk-forward 2019-26", **j))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "rules"
    cache = ROOT / "data/research/session_vp/features.parquet"
    if part == "build":
        d = features(wf.load(config.CORE_SYMBOLS))
        d.to_parquet(cache); print("features", len(d), d[VP].notna().mean().round(2).to_dict())
    else:
        d = pd.read_parquet(cache)
        for c in wf.CATEGORICAL:
            d[c] = d[c].astype("category")
        t = rules(d) if part == "rules" else ml(d)
        t.to_csv(ROOT / f"data/research/session_vp_{part}.csv", index=False)
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
        print(t.to_string(index=False))
