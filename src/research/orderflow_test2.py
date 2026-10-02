r"""
ORDERFLOW TEST 2: naye "asli orderflow" measurements (pehle test ke 10 ke alawa).

Sab kuch fill se PEHLE ka data (no lookahead). Direction-adjusted: + = flow trade ki direction mein.

ZONE PRICE PE (raw futures trades, sirf zone candle + displacement ke time pe, jab tak BOS confirm hua):
  of_zp_delta      : zone ke PRICE RANGE ke andar hue trades ka delta (buy-sell)/(buy+sell)
  of_zp_vol        : zone price range ke andar volume per minute / last 24h avg volume per minute
  of_zp_big_share  : zone price pe bade orders (>=20 contracts) ka share / last 24h ka normal share
  of_zp_big_delta  : zone price pe bade orders ka delta
  Futures price spot se alag hota hai (basis ~30-40 pip; 6J = 1/USDJPY) -> zone ko futures price mein
  convert karte hain: ratio = futures close / spot close (zone ban'ne ke waqt).

DELTA DIVERGENCE / EXHAUSTION (1-min bars, fill se pehle):
  of_delta_shift   : delta last 30 min - delta usse pehle ke 30 min (+ = against wala zor thak raha)
  of_div_30m       : price last 30 min mein zone ki taraf aaya (ATR) * of_delta_shift (dono saath = divergence)

DISPLACEMENT IMBALANCE (1-min bars, zone candle close -> BOS candle close):
  of_disp_stack    : sabse lambi lagatar minutes jisme delta trade direction mein >= 0.2 tha
  of_disp_imb_share: displacement ke active minutes mein se kitne % mein delta direction mein >= 0.2

Test (pass rule PEHLE se fix):
  5 seeds, learn 2024 / test 2025-26, rule = worst 20% drop (side 2024 se), dono baar:
  (1) bina ACTIVITY filter, (2) ACTIVITY filter ke baad (tick volume last 15 min / 24h avg >= threshold 2019-20 se).
  PASS = har seed pe PF badhe, avg +0.05 ya zyada, teeno pairs + dono saal behtar, total R 15% se zyada na gire.
  ML: base vs base + naye features (test 2025-26, 3 seeds).
Output: data/research/orderflow_test2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import orderflow as of  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402
from src.research import orderflow_test as ot  # noqa: E402

DATA = ROOT / "data" / "processed"
RESEARCH = ROOT / "data" / "research"
NEW = ["of_zp_delta", "of_zp_vol", "of_zp_big_share", "of_zp_big_delta",
       "of_delta_shift", "of_div_30m", "of_disp_stack", "of_disp_imb_share"]
SEEDS = [1, 2, 3, 4, 5]
ACT_LEARN_END = pd.Timestamp("2021-01-01", tz="UTC")


def _utc_naive(s) -> np.ndarray:
    return pd.DatetimeIndex(s).tz_convert("UTC").tz_localize(None).values.astype("datetime64[ns]")


# ----------------------------------------------------------------------
def add_activity(d: pd.DataFrame) -> pd.DataFrame:
    """tick volume last 15 min (3 M5 bars) per bar / last 24h (288 bars) per bar."""
    parts = []
    for s, g in d.groupby(d["symbol"].astype(str)):
        v = pd.read_parquet(DATA / f"{s}_M5.parquet", columns=["volume"])
        cv = np.concatenate([[0], np.cumsum(v["volume"].to_numpy(float))])
        i = v.index.searchsorted(g["fill_time"])
        a15, a24 = np.clip(i - 3, 0, None), np.clip(i - 288, 0, None)
        base = (cv[i] - cv[a24]) / np.maximum(i - a24, 1)
        g = g.copy()
        g["act_15m"] = np.where(base > 0, ((cv[i] - cv[a15]) / 3) / base, np.nan)
        parts.append(g)
    return pd.concat(parts)


def add_bar_features(d: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """delta shift / divergence / displacement imbalance from 1-min bars."""
    fut, sign = of.FUT_FOR[symbol]
    b = of.load_bars(fut)
    w = of._Window(b)
    ok = (d["fill_time"] >= b.index.min().tz_localize("UTC") + pd.Timedelta(days=1)) if b.index.tz is None else \
         (d["fill_time"] >= b.index.min() + pd.Timedelta(days=1))
    x = d.loc[ok]
    if x.empty:
        return d
    dirn = np.where(x["direction"].astype(str) == "bull", 1.0, -1.0) * sign
    fill = _utc_naive(x["fill_time"])
    m30 = np.timedelta64(30, "m")
    b1, s1, *_ = w.sums(fill - m30, fill)
    b0, s0, *_ = w.sums(fill - 2 * m30, fill - m30)
    shift = (of._norm_delta(b1, s1) - of._norm_delta(b0, s0)) * dirn

    # price move toward zone (against trade) last 30 min, futures, ATR units
    _, _, _, _, ia, ib = w.sums(fill - m30, fill)
    n = len(w.close)
    ia_c, ib_c = np.clip(ia, 0, n - 1), np.clip(ib - 1, 0, n - 1)
    same = (w.inst[ia_c] == w.inst[ib_c]) & (ib > ia)
    ret = np.log(w.close[ib_c] / w.close[ia_c]) * sign * x["entry"].to_numpy(float) / x["atr"].to_numpy(float)
    ret = np.where(same, ret, np.nan) * np.where(x["direction"].astype(str) == "bull", 1, -1)
    div = np.maximum(-ret, 0) * shift

    # displacement imbalance per minute
    tf = pd.to_timedelta(x["zone_tf"].astype(str).map(of.TF_MIN).fillna(15).to_numpy(), unit="m").values
    za = _utc_naive(x["zone_candle_time"]) + tf
    zb = _utc_naive(x["bos_time"]) + tf
    ia, ib = w.idx(za), w.idx(zb)
    buy = np.diff(w.buy)        # per-minute values
    sell = np.diff(w.sell)
    tot = buy + sell
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(tot > 0, (buy - sell) / tot, 0.0)
    stack = np.full(len(x), np.nan)
    share = np.full(len(x), np.nan)
    for k in range(len(x)):
        if ib[k] <= ia[k]:
            continue
        seg = imb[ia[k]:ib[k]] * dirn[k] >= 0.2
        share[k] = seg.mean()
        run = best = 0
        for v in seg:
            run = run + 1 if v else 0
            best = max(best, run)
        stack[k] = best
    d = d.copy()
    for c in ("of_delta_shift", "of_div_30m", "of_disp_stack", "of_disp_imb_share"):
        d[c] = np.nan
    d.loc[ok, "of_delta_shift"] = shift
    d.loc[ok, "of_div_30m"] = div
    d.loc[ok, "of_disp_stack"] = stack
    d.loc[ok, "of_disp_imb_share"] = share
    return d


def add_zone_price_features(d: pd.DataFrame, symbol: str, log=print) -> pd.DataFrame:
    """Raw trades inside the zone's price range, from zone candle open until BOS confirmed (created_at)."""
    fut, sign = of.FUT_FOR[symbol]
    d = d.copy()
    for c in ("of_zp_delta", "of_zp_vol", "of_zp_big_share", "of_zp_big_delta"):
        d[c] = np.nan
    b = of.load_bars(fut)
    w = of._Window(b)
    spot = pd.read_parquet(DATA / f"{symbol}_M5.parquet", columns=["close"])
    st, sc = spot.index.values.astype("datetime64[ns]") if spot.index.tz is None else _utc_naive(spot.index), \
        spot["close"].to_numpy(float)
    conv = (lambda p: 1.0 / p) if sign < 0 else (lambda p: p)  # noqa: E731

    rows = d.index[d["fill_time"] >= pd.Timestamp("2024-01-02", tz="UTC")]
    x = d.loc[rows]
    za = _utc_naive(x["zone_candle_time"])
    zb = _utc_naive(x["created_at"])
    zb = np.maximum(zb, za + np.timedelta64(5, "m"))
    # basis ratio at zone candle end (futures last 1m close / spot M5 close)
    tf = pd.to_timedelta(x["zone_tf"].astype(str).map(of.TF_MIN).fillna(15).to_numpy(), unit="m").values
    zend = za + tf
    fi = np.clip(w.idx(zend) - 1, 0, len(w.close) - 1)
    si = np.clip(np.searchsorted(st, zend, side="left") - 1, 0, len(sc) - 1)
    ratio = w.close[fi] / conv(sc[si])
    p1, p2 = conv(x["bottom"].to_numpy(float)) * ratio, conv(x["top"].to_numpy(float)) * ratio
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    dirn = np.where(x["direction"].astype(str) == "bull", 1.0, -1.0) * sign

    # 24h baseline from 1-min bars (per active minute; big share)
    bb, ss, bbb, bbs, ja, jb = w.sums(zend - np.timedelta64(24, "h"), zend)
    base_vpm = np.where(jb - ja >= 60, (bb + ss) / np.maximum(jb - ja, 1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        base_big = np.where(bb + ss > 0, (bbb + bbs) / (bb + ss), np.nan)

    out = {c: np.full(len(x), np.nan) for c in ("of_zp_delta", "of_zp_vol", "of_zp_big_share", "of_zp_big_delta")}
    month = pd.DatetimeIndex(za).to_period("M")
    files = {p.stem.split("_")[1]: p for p in sorted(of.RAW.glob(f"{fut}_*.parquet"))}
    for m in sorted(set(month)):
        idx = np.where(month == m)[0]
        keys = [str(m - 1), str(m), str(m + 1)]
        parts = [pd.read_parquet(files[k], columns=["ts_event", "price", "size", "side"]) for k in keys if k in files]
        if not parts:
            continue
        t = pd.concat(parts)
        t = t[t["side"].isin(["B", "A"])]
        tt = _utc_naive(t["ts_event"])
        order = np.argsort(tt, kind="stable")
        tt = tt[order]
        pr = t["price"].to_numpy(float)[order]
        sz = t["size"].to_numpy(np.int64)[order]
        sgn = np.where(t["side"].to_numpy()[order] == "B", 1, -1)
        for k in idx:
            a, e = np.searchsorted(tt, za[k]), np.searchsorted(tt, zb[k])
            if e <= a:
                continue
            inside = (pr[a:e] >= lo[k]) & (pr[a:e] <= hi[k])
            if not inside.any():
                out["of_zp_vol"][k] = 0.0
                continue
            s_, z_ = sgn[a:e][inside], sz[a:e][inside]
            vol = z_.sum()
            out["of_zp_delta"][k] = (s_ * z_).sum() / vol * dirn[k]
            mins = max((zb[k] - za[k]) / np.timedelta64(1, "m"), 1)
            out["of_zp_vol"][k] = vol / mins / base_vpm[k] if base_vpm[k] > 0 else np.nan
            big = z_ >= of.BIG
            bv = z_[big].sum()
            out["of_zp_big_share"][k] = (bv / vol) / base_big[k] if base_big[k] > 0 else np.nan
            out["of_zp_big_delta"][k] = (s_[big] * z_[big]).sum() / bv * dirn[k] if bv > 0 else 0.0
        del t, tt, pr, sz, sgn
    for c, v in out.items():
        d.loc[rows, c] = v
    log(f"{symbol}: zone-price features for {len(x):,} trades (coverage {np.isfinite(out['of_zp_vol']).mean():.0%})")
    return d


def build(symbols=("EURUSD", "USDJPY", "GBPUSD"), log=print) -> pd.DataFrame:
    d = ot.load_with_of(list(symbols), log)
    parts = []
    for s, g in d.groupby(d["symbol"].astype(str)):
        g = add_bar_features(g, s)
        g = add_zone_price_features(g, s, log)
        parts.append(g)
    d = pd.concat(parts)
    d = add_activity(d).sort_values("fill_time").reset_index(drop=True)
    return d


# ----------------------------------------------------------------------
def _r(b):
    return pd.Series(np.where(b["exit_ml"] == "trail", b["r_trail"], b["r_tp1"]), index=b.index)


def rules_test(d: pd.DataFrame, log=print) -> dict:
    P0, E0 = dict(wf.PARAMS), dict(wf.EXIT_PARAMS)
    res = {"no_activity": {}, "with_activity": {}}
    syms = ["EURUSD", "USDJPY", "GBPUSD"]
    for seed in SEEDS:
        wf.PARAMS, wf.EXIT_PARAMS = dict(P0, random_state=seed), dict(E0, random_state=seed)
        w = wf.walk_forward(d, first_test_year=2019, log=lambda *a: None)
        b = ot.bot_sel(w).copy()
        b["r"] = _r(b)
        act_thr = b.loc[b["fill_time"] < ACT_LEARN_END, "act_15m"].quantile(0.2)
        for mode in res:
            bb = b if mode == "no_activity" else b[~(b["act_15m"] < act_thr)]
            L = bb[(bb["fill_time"] >= ot.OF_START) & (bb["fill_time"] < ot.TEST_START)]
            X = bb[bb["fill_time"] >= ot.TEST_START]
            base = ot.stats(X["r"])
            res[mode].setdefault("baseline", []).append(dict(seed=seed, **base))
            for f in NEW:
                q = L[f].dropna()
                if len(q) < 200:
                    continue
                edges = q.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
                qpf = [ot.stats(L["r"][np.digitize(L[f], edges) == k])["pf"] or 0 for k in range(5)]
                lo_bad = qpf[0] <= qpf[4]
                thr = edges[0] if lo_bad else edges[3]
                keep = ~(X[f] < thr) if lo_bad else ~(X[f] > thr)
                k = X[keep]
                st = ot.stats(k["r"])
                res[mode].setdefault(f, []).append(dict(
                    seed=seed, drop="low" if lo_bad else "high", thr=round(float(thr), 4),
                    pf=st["pf"], trades=st["trades"], total_r=st["total_r"], base_pf=base["pf"], base_r=base["total_r"],
                    pairs={s: (ot.stats(X["r"][X["symbol"].astype(str) == s])["pf"],
                               ot.stats(k["r"][k["symbol"].astype(str) == s])["pf"]) for s in syms},
                    years={y: (ot.stats(X["r"][X["fill_time"].dt.year == y])["pf"],
                               ot.stats(k["r"][k["fill_time"].dt.year == y])["pf"]) for y in (2025, 2026)}))
        log(f"seed {seed} done")
    wf.PARAMS, wf.EXIT_PARAMS = P0, E0

    summary = {}
    for mode, dd in res.items():
        summary[mode] = {}
        for f in NEW:
            runs = dd.get(f)
            if not runs:
                continue
            gains = [r["pf"] - r["base_pf"] for r in runs]
            r_drop = np.mean([1 - r["total_r"] / r["base_r"] for r in runs])
            pairs_ok = all(r["pairs"][s][1] > r["pairs"][s][0] for r in runs for s in syms)
            years_ok = all(r["years"][y][1] > r["years"][y][0] for r in runs for y in (2025, 2026))
            passed = min(gains) > 0 and np.mean(gains) >= 0.05 and pairs_ok and years_ok and r_drop <= 0.15
            summary[mode][f] = dict(avg_gain=round(float(np.mean(gains)), 3), min_gain=round(float(min(gains)), 3),
                                    seeds_better=int(sum(g > 0 for g in gains)), avg_r_drop_pct=round(100 * r_drop, 1),
                                    pairs_all_better=pairs_ok, years_all_better=years_ok, PASS=bool(passed),
                                    drop_side=runs[0]["drop"])
    return dict(runs=res, summary=summary)


def ml_test(d: pd.DataFrame, log=print) -> dict:
    P0, E0, F0 = dict(wf.PARAMS), dict(wf.EXIT_PARAMS), list(wf.FEATURES)
    out = []
    for seed in SEEDS[:3]:
        wf.PARAMS, wf.EXIT_PARAMS = dict(P0, random_state=seed), dict(E0, random_state=seed)
        row = {"seed": seed}
        for name, feats in (("base", F0), ("base+new", F0 + NEW)):
            wf.FEATURES = feats
            w = wf.walk_forward(d, first_test_year=2025, log=lambda *a: None)
            b = ot.bot_sel(w)
            row[name] = ot.stats(_r(b))
        out.append(row)
        log(f"ML seed {seed}: base {row['base']['pf']} ({row['base']['trades']})  "
            f"+new {row['base+new']['pf']} ({row['base+new']['trades']})")
    wf.PARAMS, wf.EXIT_PARAMS, wf.FEATURES = P0, E0, F0
    return {"runs": out}


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    cache = RESEARCH / "orderflow_test2_data.parquet"
    if part in ("build", "all") or not cache.exists():
        d = build()
        d.to_parquet(cache)
    d = pd.read_parquet(cache)
    res_file = RESEARCH / "orderflow_test2.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    if part in ("rules", "all"):
        res["rules"] = rules_test(d)
        for mode, s in res["rules"]["summary"].items():
            print(f"\n== RULES {mode} ==")
            for f, v in s.items():
                print(f"  {f:18s} {v}")
    if part in ("ml", "all"):
        res["ml"] = ml_test(d)
    res_file.write_text(json.dumps(res, indent=1, default=str))
    print("saved -> data/research/orderflow_test2.json")
