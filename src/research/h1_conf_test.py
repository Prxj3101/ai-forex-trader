"""
H1 big-swing confluence test (16 Oct 2026, his "kr"). Same as the H4 confluence (confluence_test.py h4_zone), but H1.
RULES (fixed before the run):
  h1_zone(n): M15 zone overlaps a same-direction alive H1 BOS zone (smc.find_orderflow_zones on H1, swing n = 5 and 3,
              disp per pair), alive from created_at until an H1 close beyond its wick (max 1440 H1 bars) or 84 days.
  Measured on the bot's out-of-sample trades (confluence/wf_base_h4flag.parquet: M15 touch 1, ML top 30%, VA, activity,
  exit brain, 2019-26) and raw M15 touch 1 (no ML), split 2019-21 / 2022-26. Compared with h4_zone (live CONFLUENCE).
  Pass (as an extra / replacement confluence): bot PF > 1.05 in BOTH periods, swing 3 neighbour > 1, random-subset p < 0.05.
Run: python src/research/h1_conf_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.features.smc import find_orderflow_zones, track_zones  # noqa: E402
from src.research.zone_dataset import load  # noqa: E402

CONF = ROOT / "data/research/confluence"
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
KEY = ["symbol", "zone_id", "touch_number"]


def flag(rows, z):
    zc = z["created_at"].to_numpy(dtype="datetime64[ns]")
    zi = z["invalidated_time"].fillna(pd.Timestamp("2100-01-01", tz="UTC")).to_numpy(dtype="datetime64[ns]")
    ze = np.minimum(zi, zc + np.timedelta64(84, "D"))
    zb, zt, zbo = (z["direction"] == "bull").to_numpy(), z["top"].to_numpy(), z["bottom"].to_numpy()
    tt = rows["fill_time"].to_numpy(dtype="datetime64[ns]")
    b, top, bot = (rows["direction"] == "bull").to_numpy(), rows["top"].to_numpy(), rows["bottom"].to_numpy()
    return np.array([((zc <= tt[i]) & (tt[i] < ze) & (zb == b[i]) & (zbo <= top[i]) & (zt >= bot[i])).any()
                     for i in range(len(rows))])


def pf(r):
    r = pd.Series(r, dtype=float)
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl > 0 else None


def main():
    parts = []
    for s in SYMS:
        c = pd.read_parquet(CONF / f"{s}_conf.parquet")
        h1 = load(s, "H1")
        for n in (5, 3):
            z = track_zones(find_orderflow_zones(h1, swing_n=n, min_disp_atr=config.disp_for(s)), h1, max_bars=1440)
            c[f"h1_n{n}"] = flag(c, z)
        parts.append(c[KEY + ["h1_n5", "h1_n3", "h4_zone", "r_tp1", "r_trail", "fill_time"]])
        print(s, "done")
    f = pd.concat(parts, ignore_index=True)
    w = pd.read_parquet(CONF / "wf_base_h4flag.parquet")
    w["symbol"] = w["symbol"].astype(str)
    b = w[(w.zone_tf == "M15") & (w.touch_number == 1) & w["keep_0.3"] & w.va_ok & w.act_ok]
    b = b.merge(f[KEY + ["h1_n5", "h1_n3"]], on=KEY, how="left").fillna({"h1_n5": False, "h1_n3": False})
    rows = []
    rng = np.random.default_rng(3)

    def add(name, g):
        sims = np.array([pf(b.r_ml.sample(len(g), random_state=int(rng.integers(1e9)))) for _ in range(2000)]) if len(g) else []
        rows.append(dict(rule=name, trades=len(g), per_year=round(len(g) / 8), win=round(100 * float((g.r_ml > 0).mean()), 1),
                         pf=pf(g.r_ml), pf_19_21=pf(g.loc[g.test_year <= 2021, "r_ml"]), pf_22_26=pf(g.loc[g.test_year >= 2022, "r_ml"]),
                         total_r=round(float(g.r_ml.sum()), 1),
                         pairs=" ".join(f"{k[:3]} {pf(x.r_ml)}" for k, x in g.groupby("symbol")),
                         random_p=round(float((sims >= pf(g.r_ml)).mean()), 4) if len(g) else None))
    add("ALL bot trades", b)
    add("H4 zone (live CONFLUENCE)", b[b.h4_zone])
    add("H1 swing 5", b[b.h1_n5])
    add("H1 swing 3", b[b.h1_n3])
    add("H1 swing 5, NOT H4", b[b.h1_n5 & ~b.h4_zone])
    add("H1 swing 5 AND H4", b[b.h1_n5 & b.h4_zone])
    add("H4 OR H1 swing 5", b[b.h1_n5 | b.h4_zone])
    add("no H4, no H1 swing 5", b[~b.h1_n5 & ~b.h4_zone])
    rep = pd.DataFrame(rows)
    raw = f.assign(per=np.where(f.fill_time.dt.year <= 2020, "16-20", "21-26"))
    from src.ml.walkforward import add_costs  # noqa: F401  (raw r_tp1 here is gross; only YES vs NO compared)
    pd.set_option("display.width", 250)
    print(rep.to_string(index=False))
    print("\nRAW (no ML, gross 1:1) YES vs NO:")
    for c_ in ("h1_n5", "h1_n3", "h4_zone"):
        print(" ", c_, {p: (pf(g.loc[g[c_], "r_tp1"]), pf(g.loc[~g[c_], "r_tp1"])) for p, g in raw.groupby("per")},
              f"yes {100 * raw[c_].mean():.1f}%")
    rep.to_csv(CONF / "h1_conf_test.csv", index=False)


if __name__ == "__main__":
    main()
