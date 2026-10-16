"""
Zone life test (16 Oct 2026) - kya PURANE untapped zones bhi kaam karte hain?

Abhi: zone sirf LIFE_DAYS = 5 trading din tak zinda (zone_dataset_v2 + live_features).
Prajwal ke examples (USDJPY 1h: 24 Sep -> 14 Oct tap, EURUSD 15m: 9 Oct -> 15 Oct) dikhate hain ki
purane zones bhi respect hote hain. Ye test zone life badha ke dekhta hai.

RULES (run se PEHLE fix):
  - Lives: 5 (baseline), 10, 20, 40 trading din. 40 = "jab tak tap ya break na ho" (practically).
  - Baaki sab same: entry, SL, max 3 touches, M5 close wick ke paar = zone dead, costs $5 + 0.2 pip.
  - Sab lives M5 labels pe (tick replay nahi) -> apples to apples. Baseline bhi M5 pe banta hai.
  - Verdict: life X tabhi better jab
      (a) ML walk-forward bot (M15 touch1 + top 30% + exit brain, 2019-26) ka PF baseline se >= ho, AUR
      (b) sirf PURANE fills (age > 5 din) ka raw PF (touch 1, sab trades) >= naye fills (0-5 din) ka ~ 0.95x, AUR
      (c) 2021-26 alag se bhi same direction.
  - Output: data/research/zone_life/life{N}/{SYM}_v2.parquet + report.csv. Koi model / live file overwrite nahi.

Chalana:
  python src/research/zone_life_test.py build --life 10 --symbol EURUSD
  python src/research/zone_life_test.py report
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import zone_dataset_v2 as zd  # noqa: E402

OUT = ROOT / "data" / "research" / "zone_life"
LIVES = [5, 10, 20, 40]
SYMS = ["EURUSD", "USDJPY", "GBPUSD"]
AGE_BINS = [0, 5, 10, 20, 40]      # trading din


def build(life: int, sym: str) -> Path:
    zd.LIFE_DAYS = life
    ds = zd.build(sym, tick_labels=False, log=lambda a, m, p=None: None)
    f = OUT / f"life{life}" / f"{sym}_v2.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(f)
    print(f"life {life} {sym}: {len(ds):,} trades -> {f.relative_to(ROOT)}")
    return f


def pf(r):
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def stats(g, col="r_tp1"):
    return dict(trades=len(g), pf=pf(g[col]), avg_r=round(float(g[col].mean()), 3) if len(g) else np.nan,
                total_r=round(float(g[col].sum()), 1))


def report(skip_ml: bool = False, lives=None) -> None:
    from src.ml import walkforward as wf
    rows = []
    for life in (lives or LIVES):
        folder = OUT / f"life{life}"
        if not all((folder / f"{s}_v2.parquet").exists() for s in SYMS):
            print(f"life {life}: dataset missing, skip")
            continue
        d = wf.load(SYMS, folder=folder)
        d["age_days"] = d["bars_since_created"] / zd.BARS_PER_DAY
        d["age"] = pd.cut(d["age_days"], AGE_BINS, right=False, labels=["0-5", "5-10", "10-20", "20-40"])
        t1 = d[d["touch_number"] == 1]
        # RAW (no ML): touch 1, by zone tf x age, both periods
        for per, m in (("2016-26", t1["test_year"] >= 0), ("2021-26", t1["test_year"] >= 2021)):
            for tf in ("M15", "H1"):
                for age, g in t1[m & (t1["zone_tf"] == tf)].groupby("age", observed=True):
                    rows.append(dict(life=life, kind="raw touch1", period=per, tf=tf, age=str(age), **stats(g)))
        if skip_ml:
            continue
        w = wf.walk_forward(d, 2019, log=lambda *a: None)
        res = wf.results(w)
        b = res["bot"]
        rows.append(dict(life=life, kind="ML bot", period="2019-26", tf="M15", age="all", trades=b["trades"],
                         pf=b["pf"], avg_r=b["avg_r"], total_r=b["total_r"]))
        w["age_days"] = w["bars_since_created"] / zd.BARS_PER_DAY
        sel = w[(w["zone_tf"] == "M15") & (w["touch_number"] == 1) & w["keep_0.3"] & w["va_ok"] & w["act_ok"]]
        for per, m in (("2019-26", sel["test_year"] >= 0), ("2021-26", sel["test_year"] >= 2021)):
            s = sel[m]
            rows.append(dict(life=life, kind="ML bot", period=per, tf="M15", age="all", **stats(s, "r_ml")))
            for lab, mm in (("0-5", s["age_days"] < 5), ("5+", s["age_days"] >= 5)):
                rows.append(dict(life=life, kind="ML bot", period=per, tf="M15", age=lab, **stats(s[mm], "r_ml")))
        # H1 touch1 top30 bhi (bot sirf M15 leta hai, lekin dekhna hai)
        h1 = w[(w["zone_tf"] == "H1") & (w["touch_number"] == 1) & w["keep_0.3"]]
        for lab, mm in (("0-5", h1["age_days"] < 5), ("5+", h1["age_days"] >= 5)):
            rows.append(dict(life=life, kind="ML top30 H1", period="2019-26", tf="H1", age=lab, **stats(h1[mm], "r_ml")))
        print(f"life {life}: done")
    rep = pd.DataFrame(rows)
    name = "report_raw" if skip_ml else "report"
    if lives:
        name += "_" + "_".join(map(str, lives))
    rep.to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--life", type=int, nargs="+", default=LIVES)
    b.add_argument("--symbol", nargs="+", default=SYMS)
    r = sp.add_parser("report")
    r.add_argument("--life", type=int, nargs="+", default=None)
    r.add_argument("--raw", action="store_true", help="skip ML (fast)")
    a = ap.parse_args()
    if a.cmd == "build":
        for life in a.life:
            for s in a.symbol:
                build(life, s)
    else:
        report(a.raw, a.life)


if __name__ == "__main__":
    main()
