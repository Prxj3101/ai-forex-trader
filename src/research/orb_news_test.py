r"""
ORB NEWS BLOCK TEST (15 Oct 2026). His question after the USDJPY ORB ASIA buy filled at 08:30:01 NY (CPI second):
should ORB skip breaks around high-impact news (like the zone bot, -5/+20 min)?
DATA   1,417 live-rule ORB trades 2019-26 (data/research/orb_risk_cands.parquet: tick fills/exits, net R incl. $5/lot + 0.2 pip
       stop slippage). News = data/news/forex_factory_calendar.csv, HIGH impact, currencies of the pair (EURUSD: EUR+USD, ...).
RULES  (fixed before running) a trade is BLOCKED if its fill time is inside [news - before, news + after].
       PRIMARY = -5/+20 (zone bot rule). Also -5/+5, -5/+10, -5/+30, -15/+60.
STRESS real stop orders at news slip more than ticks show: news trades (fill within -5/+20) get extra entry slippage
       0 / 1 / 2 / 3 pips -> R reduced by slip/risk.
Chalana: python src\research\orb_news_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CUR = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY")}


def pf(s):
    s = pd.Series(s).dropna()
    return round(float(s[s > 0].sum() / -s[s < 0].sum()), 3) if (s < 0).any() else np.nan


def main():
    t = pd.read_parquet(ROOT / "data/research/orb_risk_cands.parquet")
    t["fill"] = pd.to_datetime(t.fill_time, utc=True)
    n = pd.read_csv(ROOT / "data/news/forex_factory_calendar.csv", usecols=["DateTime", "Currency", "Impact", "Event"])
    n = n[n.Impact.str.contains("High", na=False)].copy()
    n["t"] = pd.to_datetime(n.DateTime, utc=True, format="mixed")
    n = n[(n.t >= "2018-12-01") & ~n.Event.str.contains("Bank Holiday", na=False)]
    ev = {c: np.sort(g.t.to_numpy().astype("datetime64[s]").astype("int64")) for c, g in n.groupby("Currency")}
    fs = t.fill.to_numpy().astype("datetime64[s]").astype("int64")

    def near(before, after):
        out = np.zeros(len(t), bool)
        for i, (sym, f) in enumerate(zip(t.symbol, fs)):
            for c in CUR[sym]:
                e = ev.get(c)
                if e is None:
                    continue
                j = np.searchsorted(e, f - after * 60)          # first event >= fill - after
                if j < len(e) and e[j] <= f + before * 60:
                    out[i] = True
        return out

    R = t.r_net_bt
    half = t.fill.dt.year <= 2022
    print(f"ORB trades {len(t)}: PF {pf(R)}  +{R.sum():.1f}R  | 2019-22 PF {pf(R[half])} | 2023-26 PF {pf(R[~half])}")
    rows = []
    for b, a in [(5, 5), (5, 10), (5, 20), (5, 30), (15, 60)]:
        m = near(b, a)
        k = R[~m]
        rows.append(dict(block=f"-{b}/+{a}", news_trades=int(m.sum()), news_PF=pf(R[m]), news_R=round(R[m].sum(), 1),
                         kept=int((~m).sum()), kept_PF=pf(k), kept_R=round(k.sum(), 1),
                         kept_PF_19_22=pf(R[~m & half]), kept_PF_23_26=pf(R[~m & ~half])))
    print(pd.DataFrame(rows).to_string(index=False))
    m = near(5, 20)
    pip = np.where(t.symbol.str.endswith("JPY"), 0.01, 0.0001)
    print("\nSTRESS: extra entry slippage on news trades (-5/+20) vs blocking them")
    for s in [0, 1, 2, 3]:
        r2 = R - np.where(m, s * pip / t.risk, 0.0)
        print(f"  slip {s} pip: no block PF {pf(r2)} ({r2.sum():+.1f}R) | block PF {pf(R[~m])} ({R[~m].sum():+.1f}R)")
    x = t[m].assign(R=R[m])
    print("\nnews trades by pair/range:", x.groupby(["symbol", "range"]).R.agg(["count", "sum"]).round(1).to_dict("index"))


if __name__ == "__main__":
    main()
