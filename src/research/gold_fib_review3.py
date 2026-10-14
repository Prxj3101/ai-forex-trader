r"""
FIB 7 REVIEW ROUND 3 (14 Oct 2026) - labels for the ML model ("learn Prajwal's eye").
Pool: V3 zones with zigzag K = 2.5 (bigger swings; also includes smaller ones so the model learns his "too small"/"not clean"),
H1 + M15, created 2016-2020 ONLY (2021-26 stays untouched for the one final test), first-tap fill (FT2 = first chart touch).
300 setups (150 H1 + 150 M15, seed fixed), ordered so every batch of 50 has 25 H1 + 25 M15. Setups that match a round-2
setup (same TF, created within 2 bars) are dropped, and so are zones whose 0.705 was already touched on M5 before the
fill (inside the move-end bar - fresh check bug in v2/v3, ~3% of zones), and zones where price went past fib 0 before
the fill (new move = old fib dead; this rule was MISSING in v1-v3 - ~40% of zones). Results stay hidden from him.
Writes data/research/gold_fib/review3/key.parquet (all zone columns + r_* results, for the ML) + data/hNNN.json + meta.json.
Chart shows SL A 0.816 and TP 1:1 like round 2 (labels comparable).
Chalana: python src\research\gold_fib_review3.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/research/gold_fib_v3"
OUT = ROOT / "data/research/gold_fib/review3"
R2 = ROOT / "data/research/gold_fib/review2/data"
SEED = 20261015
N = 150
N_BARS = {"M5": 300, "M15": 300, "H1": 250, "H4": 150}
LEVELS = [0, 0.5, 0.618, 0.705, 0.786, 0.886, 1]
BAR = {"H1": 3600, "M15": 900}


def tapped_before(m5, r):
    """True if an M5 bar that CLOSED after the move end and before the fill already touched 0.705 (e.g. inside the
    move-end bar - the H1/M15 fresh check can't see the order inside one bar), OR price went past fib 0 (new move =
    old fib dead, his rule "valid jab tak naya move na bane" - missing in v1-v3: ~40% of zones!)."""
    t0 = r.t_end
    t1 = pd.Timestamp(int(r.fill_FT2), unit="ms", tz="UTC")
    x = m5[(m5.index > t0) & (m5.close_time <= t1)]
    if r.bull:
        return bool((x.low <= r.entry).any() or (x.high > r.H).any())
    return bool((x.high >= r.entry).any() or (x.low < r.L).any())


def write(pick, key_path, meta_path):
    bars = {tf: pd.read_parquet(ROOT / f"data/processed/XAUUSD_{tf}.parquet", columns=["open", "high", "low", "close", "close_time"])
            for tf in N_BARS}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    meta = []
    for _, r in pick.iterrows():
        bull = bool(r.bull)
        lvl = {str(f): float(r.H - f * (r.H - r.L) if bull else r.L + f * (r.H - r.L)) for f in LEVELS}
        e = float(r.entry)
        risk = abs(e - r["sl_A_0.816"])
        tfd = {}
        for tf, b in bars.items():
            x = b[b.close_time <= r.fill].tail(N_BARS[tf])
            tfd[tf] = [[int(t.timestamp()), round(o, 2), round(h, 2), round(lo, 2), round(c, 2)]
                       for t, o, h, lo, c in zip(x.index, x.open, x.high, x.low, x.close)]
        obj = dict(id=r.id, move_tf=r.tf, side="BUY" if bull else "SELL", digits=2,
                   t_start=int(r.t_start.timestamp()), t_end=int(r.t_end.timestamp()), created=int(r.created.timestamp()),
                   fill=int(r.fill.timestamp()), L=round(float(r.L), 2), H=round(float(r.H), 2), levels=lvl,
                   entry=round(e, 2), sl=round(float(r["sl_A_0.816"]), 2), tp=round(e + risk if bull else e - risk, 2),
                   bos=bool(r.bos), session=bool(r.session), sweep=bool(r.sweep), tf=tfd)
        (OUT / "data" / f"{r.id}.json").write_text(json.dumps(obj))
        meta.append(dict(id=r.id, move_tf=r.tf, side=obj["side"], move_usd=round(float(r["size"]), 2), sl_usd=round(risk, 2),
                         bos=obj["bos"], session=obj["session"], sweep=obj["sweep"],
                         when=r.fill.tz_convert("America/New_York").strftime("%a %H:%M NY")))
    pick.to_parquet(key_path)
    meta_path.write_text(json.dumps(meta))


def main():
    m5 = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"])
    r2 = [json.loads(p.read_text()) for p in sorted(R2.glob("g*.json"))]
    r2 = [(o["move_tf"], o["created"]) for o in r2]
    picks = {}
    for tf in ["H1", "M15"]:
        d = pd.read_parquet(SRC / f"K2.5_{tf}.parquet")
        d = d[d.fill_FT2.notna() & (d.created.dt.year <= 2020)].copy()
        cs = d.created.astype("int64") // 10**9
        clash = np.zeros(len(d), bool)
        for t, c in r2:
            if t == tf:
                clash |= (abs(cs - c) <= 2 * BAR[tf]).to_numpy()
        d = d[~clash]
        d = d[[not tapped_before(m5, r) for _, r in d.iterrows()]]       # inner-bar tap (M5) before the first tap -> not fresh
        picks[tf] = d.sample(N, random_state=SEED).reset_index(drop=True)
    order = []
    for k in range(N):
        order += [picks["H1"].iloc[k], picks["M15"].iloc[k]]
    rng = np.random.default_rng(SEED)
    pick = pd.DataFrame(order).reset_index(drop=True)
    idx = np.concatenate([rng.permutation(np.arange(b, b + 50)) for b in range(0, 2 * N, 50)])
    pick = pick.iloc[idx].reset_index(drop=True)
    pick["id"] = [f"h{i + 1:03d}" for i in range(len(pick))]
    pick["fill"] = pd.to_datetime(pick.fill_FT2.astype("int64"), unit="ms", utc=True)
    write(pick, OUT / "key.parquet", OUT / "meta.json")
    print(len(pick), "setups;", pick.groupby((pick.index // 50)).tf.value_counts().unstack().to_string())


def extra(n_new=50, seed=SEED + 1):
    """15 Oct: he + Harsh trade 15m only (his H1 YES/NO showed no edge: 'H1 ka krte hi nhi'). Add n_new MORE M15
    setups (h301..) from the same pool, not already picked, same filters. Writes key_extra.parquet + data/h3NN.json."""
    m5 = pd.read_parquet(ROOT / "data/processed/XAUUSD_M5.parquet", columns=["high", "low", "close_time"])
    k = pd.read_parquet(OUT / "key.parquet")
    r2 = [json.loads(p.read_text()) for p in sorted(R2.glob("g*.json"))]
    r2 = [(o["move_tf"], o["created"]) for o in r2 if o["move_tf"] == "M15"]
    d = pd.read_parquet(SRC / "K2.5_M15.parquet")
    d = d[d.fill_FT2.notna() & (d.created.dt.year <= 2020)].copy()
    cs = d.created.astype("int64") // 10**9
    clash = np.zeros(len(d), bool)
    for _, c in r2:
        clash |= (abs(cs - c) <= 2 * BAR["M15"]).to_numpy()
    used = set(k[k.tf == "M15"].created.astype("int64"))
    d = d[~clash & ~d.created.astype("int64").isin(used)]
    d = d[[not tapped_before(m5, r) for _, r in d.iterrows()]]
    pick = d.sample(n_new, random_state=seed).reset_index(drop=True)
    pick["id"] = [f"h{301 + i:03d}" for i in range(len(pick))]
    pick["fill"] = pd.to_datetime(pick.fill_FT2.astype("int64"), unit="ms", utc=True)
    write(pick, OUT / "key_extra.parquet", OUT / "meta_extra.json")
    print(len(pick), "extra M15 setups")


if __name__ == "__main__":
    import sys
    extra() if len(sys.argv) > 1 and sys.argv[1] == "extra" else main()
