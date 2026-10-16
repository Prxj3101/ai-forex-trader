"""
H4 old zones + current zone bot ML (16 Oct 2026). Prajwal: "alag bot nahi, abhi wale zone bot mein hi daalo,
ML + filters sab lagake best combo dekho".

Tareeka (overfit se bachne ke liye, rules run se PEHLE fix):
  1. H4 zones (smc.find_orderflow_zones, BOS swing n = 5 aur 3, disp per pair) bot ke HI pipeline se
     (zone_dataset_v2.zones_for + simulate_touches + VP + activity features), life 60 trading din, M5 labels.
  2. Abhi ke M15 + H1 data (zone_life/life5, same M5 labels) ke saath jod ke, SAME walk-forward ML
     (walkforward.walk_forward, 2019-26 har saal naya model, sirf pichle data pe train). zone_tf = "H4" naya category.
  3. Report sirf out-of-sample: H4 touch 1, saare / age 5+ din, ML keep 100 / 70 / 50 / 30 %, VA + activity filter,
     exits 1:1 / 1:2 / trail / exit brain. Periods 2019-21 aur 2022-26 alag.
  4. Bot + H4 combined: abhi ka bot (M15 touch1 top30 + filters + exit brain) + H4 rule, PF aur R kitna badla.
  5. Pass: H4 rule NET PF > 1.05 dono periods mein, aur bot + H4 ka PF bot akele se behtar.
Chalana: python src/research/h4_merge_test.py build --symbol EURUSD --n 5 ; ... ; python src/research/h4_merge_test.py report --n 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.features.activity import add_activity  # noqa: E402
from src.features.smc import find_orderflow_zones, structure_breaks, track_zones  # noqa: E402
from src.features.volume_profile import add_vp_features  # noqa: E402
from src.research import zone_dataset_v2 as zd  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402

OUT = ROOT / "data" / "research" / "zone_h4"
BASE = ROOT / "data" / "research" / "zone_life" / "life5"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]


def build(sym, n):
    zd.LIFE_DAYS = 60
    m5, m15, h1, h4, d1 = (load(sym, tf) for tf in ["M5", "M15", "H1", "H4", "D1"])
    h1_zones = track_zones(find_orderflow_zones(h1, swing_n=config.SWING_N, min_disp_atr=1.5), h1)
    bias_h4, bias_d1 = structure_breaks(h4, 2), structure_breaks(d1, 2)
    z = zd.zones_for(h4, h1, h4, h1_zones, d1, bias_h4, bias_d1, n, config.disp_for(sym), "H4")
    t = zd.simulate_touches(z, m5)
    z = z.reset_index(drop=True)
    t = t.merge(z, left_on="zone_id", right_index=True, how="left", suffixes=("", "_zone"))
    t["zone_id"] = "H4_" + t["zone_id"].astype(str)
    t = t.sort_values("fill_time").reset_index(drop=True)
    t["symbol"], t["swing_n"], t["min_disp"], t["entry_mode"] = sym, n, config.disp_for(sym), f"{config.ENTRY_LEVEL}:body"
    t["win"] = t["r_trail"] > 0
    t = add_vp_features(t, sym, m5)
    t = add_activity(t, sym, m5)
    base = pd.read_parquet(BASE / f"{sym}_v2.parquet")
    combo = pd.concat([base, t[[c for c in t.columns if c in base.columns]]], ignore_index=True)
    combo = combo.sort_values("fill_time").reset_index(drop=True)
    f = OUT / f"n{n}" / f"{sym}_v2.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    combo.to_parquet(f)
    print(f"{sym} n{n}: {len(t):,} H4 trades + {len(base):,} base -> {f.relative_to(ROOT)}")


def pf(r):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def st(g, col):
    cum = g.sort_values("fill_time")[col].cumsum()
    return dict(trades=len(g), per_year=round(len(g) / 8, 1), pf=pf(g[col]), total_r=round(float(g[col].sum()), 1),
                max_dd_r=round(float((cum.cummax() - cum).max()), 1) if len(g) else np.nan,
                pf_19_21=pf(g.loc[g.test_year <= 2021, col]), pf_22_26=pf(g.loc[g.test_year >= 2022, col]))


def report(n):
    from src.ml import walkforward as wf
    d = wf.load(SYMS, folder=OUT / f"n{n}")
    w = wf.walk_forward(d, 2019, log=lambda *a: None)
    w["age_days"] = w["bars_since_created"] / zd.BARS_PER_DAY
    w["r_tp2"] = w["r_tp2"]
    w.to_parquet(OUT / f"wf_n{n}.parquet")
    rows = []
    h4 = w[(w.zone_tf == "H4") & (w.touch_number == 1)]
    for age, ga in (("all", h4), ("5+", h4[h4.age_days >= 5]), ("0-5", h4[h4.age_days < 5])):
        for k in (1.0, 0.7, 0.5, 0.3):
            g = ga[ga[f"keep_{k}"]]
            for filt, gf in (("none", g), ("va+act", g[g.va_ok & g.act_ok])):
                for col in ("r_tp1", "r_tp2", "r_trail", "r_ml"):
                    rows.append(dict(part=f"H4 age {age}", ml_keep=k, filters=filt, exit=col, **st(gf, col)))
    bot = w[(w.zone_tf == "M15") & (w.touch_number == 1) & w["keep_0.3"] & w.va_ok & w.act_ok]
    rows.append(dict(part="BOT now (M15)", ml_keep=0.3, filters="va+act", exit="r_ml", **st(bot, "r_ml")))
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / f"report_n{n}.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--symbol", nargs="+", default=SYMS)
    b.add_argument("--n", type=int, nargs="+", default=[5, 3])
    r = sp.add_parser("report")
    r.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "build":
        for n in a.n:
            for s in a.symbol:
                build(s, n)
    else:
        report(a.n)
