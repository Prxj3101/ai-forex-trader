r"""
SESSION VOLUME PROFILE from real ticks (Dukascopy tick count = volume proxy; forex has no central volume).

Sessions (New York time): ASIA 17:00-02:00 (incl. rollover), LONDON 02:00-08:00, NY 08:00-17:00, plus DAY
(NY 17:00 -> 17:00). Price bins = 1 pip (bid). For every profile: start/end (UTC ms), POC, VAH, VAL (70% of
volume, expanded from the POC like TradingView), and the full histogram (bins, counts) for HVN/LVN checks.
Output: data/research/session_vp/{SYMBOL}.parquet
Chalana:  python src\features\session_vp.py EURUSD 2016 2020   (years in steps, appended)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.backtest import tick_engine as te  # noqa: E402

OUT = ROOT / "data/research/session_vp"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 0.1}
NY = "America/New_York"


def value_area(bins, counts, share=0.70):
    poc_i = int(np.argmax(counts))
    total, have = counts.sum(), counts[poc_i]
    lo = hi = poc_i
    while have < share * total and (lo > 0 or hi < len(counts) - 1):
        up = counts[hi + 1] if hi < len(counts) - 1 else -1
        dn = counts[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; have += up
        else:
            lo -= 1; have += dn
    return bins[poc_i], bins[hi], bins[lo]


def build_month(sym, ym):
    z = te._month(sym, ym)
    if z is None or len(z[0]) == 0:
        return []
    t, a, b = z
    ny = pd.to_datetime(t, unit="ms", utc=True).tz_convert(NY)
    hour = ny.hour.to_numpy()
    day = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None).to_numpy()     # trading day (5 PM NY roll)
    sess = np.where((hour >= 2) & (hour < 8), 1, np.where((hour >= 8) & (hour < 17), 2, 0))   # 0 asia 1 london 2 ny
    pb = np.round(b / PIP[sym]).astype(np.int64)
    rows = []
    for kind, key in (("DAY", day.astype("datetime64[D]").astype(np.int64) * 10),
                      ("SESS", day.astype("datetime64[D]").astype(np.int64) * 10 + sess)):
        order = np.argsort(key, kind="stable")
        k_s, pb_s, t_s = key[order], pb[order], t[order]
        cuts = np.flatnonzero(np.diff(k_s)) + 1
        for seg in np.split(np.arange(len(k_s)), cuts):
            if len(seg) < 50:
                continue
            bins, counts = np.unique(pb_s[seg], return_counts=True)
            full = np.arange(bins[0], bins[-1] + 1)
            c = np.zeros(len(full), dtype=np.int64); c[bins - bins[0]] = counts
            poc, vah, val = value_area(full, c)
            kk = int(k_s[seg[0]])
            name = "DAY" if kind == "DAY" else ["ASIA", "LONDON", "NY"][kk % 10]
            rows.append(dict(symbol=sym, kind=name, day=pd.Timestamp(kk // 10, unit="D"),
                             start_ms=int(t_s[seg].min()), end_ms=int(t_s[seg].max()),
                             poc=poc * PIP[sym], vah=vah * PIP[sym], val=val * PIP[sym],
                             bin0=int(full[0]), counts=c.astype(np.int32)))
    return rows


def build(sym, y0, y1):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            rows += build_month(sym, f"{y}-{m:02d}")
    d = pd.DataFrame(rows)
    f = OUT / f"{sym}_{y0}_{y1}.parquet"
    d.to_parquet(f)
    print(sym, y0, y1, len(d), "profiles", flush=True)


def load(sym):
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob(f"{sym}_*.parquet"))], ignore_index=True)
    # a session/day split across month files -> keep the bigger piece (month boundary is mid-session rarely)
    d["n"] = d["counts"].apply(lambda c: int(np.sum(c)))
    d = d.sort_values("n").drop_duplicates(["kind", "day"], keep="last").sort_values("end_ms").reset_index(drop=True)
    return d


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
