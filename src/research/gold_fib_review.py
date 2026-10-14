r"""
FIB 7 REVIEW (14 Oct): 30 random bot setups (15 H1 + 15 M15, filled, 2021-2026, seed fixed) shown to Prajwal WITHOUT the result.
He answers per setup: "yes, I would draw this fib / take it" or "no" + why. Tells us if the bot's moves match his.
Writes data/research/gold_fib/review/key.parquet (hidden results) + data/fNN.json (charts closed at the fill moment).
ROUND 2 (14 Oct): 60 setups (30 H1 + 30 M15) from the V2 rules (big swings, fresh zone, H4 trend), years 2016-2020 ONLY
(learn period - keeps 2021-26 clean for the final check), M5 chart added (he enters on 15m / 5m).
Chalana: python src\research\gold_fib_review.py            (round 1)
         python src\research\gold_fib_review.py round2     (round 2)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/research/gold_fib"
OUT = SRC / "review"
SEED = 20261014
N_BARS = {"M5": 300, "M15": 300, "H1": 250, "H4": 150}
LEVELS = [0, 0.5, 0.618, 0.705, 0.786, 0.886, 1]


def main(round2=False):
    global OUT
    src = ROOT / "data/research/gold_fib_v2" if round2 else SRC
    if round2:
        OUT = SRC / "review2"
    d = pd.concat([pd.read_parquet(f) for f in sorted(src.glob("*_20*.parquet"))], ignore_index=True)
    d = d[(d.status == "filled")].copy()
    d["fill"] = pd.to_datetime(d.fill_ms, unit="ms", utc=True)
    d = d[d.fill.dt.year <= 2020] if round2 else d[d.fill.dt.year >= 2021]
    n = 30 if round2 else 15
    pick = pd.concat([g.sample(n, random_state=SEED) for _, g in d.groupby("tf")]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    pick["id"] = [f"{'g' if round2 else 'f'}{i + 1:02d}" for i in range(len(pick))]
    bars = {tf: pd.read_parquet(ROOT / f"data/processed/XAUUSD_{tf}.parquet", columns=["open", "high", "low", "close", "close_time"])
            for tf in N_BARS}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    meta = []
    for _, r in pick.iterrows():
        bull = bool(r.bull)
        lvl = {str(f): float(r.H - f * (r.H - r.L) if bull else r.L + f * (r.H - r.L)) for f in LEVELS}
        risk = abs(r.entry - r["sl_A_0.816"])
        tfd = {}
        for tf, b in bars.items():
            x = b[b.close_time <= r.fill].tail(N_BARS[tf])
            tfd[tf] = [[int(t.timestamp()), round(o, 2), round(h, 2), round(l, 2), round(c, 2)]
                       for t, o, h, l, c in zip(x.index, x.open, x.high, x.low, x.close)]
        obj = dict(id=r.id, move_tf=r.tf, side="BUY" if bull else "SELL", digits=2,
                   t_start=int(r.t_start.timestamp()), t_end=int(r.t_end.timestamp()), created=int(r.created.timestamp()),
                   fill=int(r.fill.timestamp()), L=round(float(r.L), 2), H=round(float(r.H), 2), levels=lvl,
                   entry=round(float(r.entry), 2), sl=round(float(r["sl_A_0.816"]), 2),
                   tp=round(float(r.entry + risk if bull else r.entry - risk), 2),
                   bos=bool(r.bos), session=bool(r.session), sweep=bool(r.sweep), tf=tfd)
        (OUT / "data" / f"{r.id}.json").write_text(json.dumps(obj))
        meta.append(dict(id=r.id, move_tf=r.tf, side=obj["side"], move_usd=round(float(r["size"]), 2), sl_usd=round(risk, 2),
                         bos=obj["bos"], session=obj["session"], sweep=obj["sweep"],
                         when=r.fill.tz_convert("America/New_York").strftime("%a %d %b %Y %H:%M NY")))
    keep = ["id", "tf", "bull", "fill", "size", "atr", "entry", "bos", "session", "sweep"] + [c for c in pick.columns if c.startswith("r_")]
    pick[keep].to_parquet(OUT / "key.parquet")
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(len(meta), "setups |", pick.tf.value_counts().to_dict(), "| BUY", int(pick.bull.sum()))


if __name__ == "__main__":
    import sys
    main(round2=len(sys.argv) > 1 and sys.argv[1] == "round2")
