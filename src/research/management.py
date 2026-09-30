r"""
Trade management test: SAME entries, alag-alag EXIT rules.

Sawaal: "Agar main trade ko alag tarike se manage karta (1:1, breakeven, trailing, partial),
         toh 10 saal ka result kaisa hota?"

Entry wahi rehti hai jo zone dataset mein hai (zone body pe limit, SL wick ke paar).
Sirf exit ka rule badalta hai. Isse fair comparison hota hai.

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\research\management.py --symbol USDJPY --tf M15

Pehle zone dataset bana hona chahiye (zone_dataset.py ya dashboard ka BUILD DATASET).

Conservative rules (backtest ko zyada achha dikhne se rokne ke liye):
  - Ek M5 candle mein pehle STOP check hota hai, phir target. (Candle ke andar kya pehle hua, pata nahi.)
  - Breakeven / trailing jis candle mein trigger ho, naya SL AGLI candle se lagta hai.
  - Long BID pe band hota hai, short ASK pe (spread ka kharcha included).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research.analysis import load_dataset  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "research"
HOLD_BARS = 3 * 288   # max 3 din (M5 candles)


@dataclass
class Policy:
    name: str
    label: str                         # dashboard/terminal ke liye simple English
    tp_r: float | None = 2.0           # target kitne R pe (None = koi fixed target nahi)
    be_trigger_r: float | None = None  # itne R profit pe SL entry pe (breakeven)
    partial_r: float | None = None     # itne R pe position ka kuch hissa band
    partial_frac: float = 0.5          # kitna hissa band (0.5 = aadha)
    trail_r: float | None = None       # trailing stop: best price se itne R peeche
    trail_start_r: float = 1.0         # trailing kab se shuru


POLICIES = [
    Policy("tp2", "Fixed 1:2 (current)", tp_r=2.0),
    Policy("tp1", "Fixed 1:1", tp_r=1.0),
    Policy("tp1_5", "Fixed 1:1.5", tp_r=1.5),
    Policy("tp3", "Fixed 1:3", tp_r=3.0),
    Policy("be1_tp2", "Breakeven at 1R, target 1:2", tp_r=2.0, be_trigger_r=1.0),
    Policy("be1_tp3", "Breakeven at 1R, target 1:3", tp_r=3.0, be_trigger_r=1.0),
    Policy("half1_be_tp2", "Close half at 1R + breakeven, rest 1:2", tp_r=2.0, be_trigger_r=1.0, partial_r=1.0),
    Policy("trail1", "Trailing stop 1R after 1R profit", tp_r=None, trail_r=1.0, trail_start_r=1.0),
    Policy("be1_trail1_tp4", "Breakeven at 1R, trail 1R, cap 1:4", tp_r=4.0, be_trigger_r=1.0, trail_r=1.0, trail_start_r=2.0),
]


def run_trade(p: Policy, bull: bool, entry: float, risk: float,
              hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, sp: np.ndarray) -> tuple[float, str]:
    """Ek trade ko bar-by-bar chalata hai. Returns (R result, exit reason)."""
    sl_r = -1.0            # SL entry se kitne R pe (shuru mein -1R)
    open_frac = 1.0        # position ka kitna hissa abhi khula hai
    realized = 0.0         # band kiye hisse ka R
    best = 0.0             # ab tak ka best favorable R
    partial_done = False

    for i in range(len(hi)):
        if bull:
            fav = (hi[i] - entry) / risk               # long BID pe band hota hai
            adv = (lo[i] - entry) / risk
        else:
            fav = (entry - (lo[i] + sp[i])) / risk     # short ASK pe band hota hai
            adv = (entry - (hi[i] + sp[i])) / risk

        # 1) STOP pehle (conservative)
        if adv <= sl_r:
            total = realized + open_frac * sl_r
            reason = "sl" if sl_r < -0.01 else ("be" if abs(sl_r) <= 0.01 else "trail")
            return total, reason

        # 2) Partial close
        if p.partial_r is not None and not partial_done and fav >= p.partial_r:
            realized += p.partial_frac * p.partial_r
            open_frac -= p.partial_frac
            partial_done = True

        # 3) Target
        if p.tp_r is not None and fav >= p.tp_r:
            return realized + open_frac * p.tp_r, "tp"

        # 4) Agli candle ke liye SL update (breakeven / trailing)
        best = max(best, fav)
        if p.be_trigger_r is not None and best >= p.be_trigger_r:
            sl_r = max(sl_r, 0.0)
        if p.trail_r is not None and best >= p.trail_start_r:
            sl_r = max(sl_r, best - p.trail_r)

    # Time khatam: last close pe band
    px = cl[-1] if bull else cl[-1] + sp[-1]
    last_r = ((px - entry) if bull else (entry - px)) / risk
    return realized + open_frac * last_r, "timeout"


def evaluate(ds: pd.DataFrame, m5: pd.DataFrame, policies: list[Policy]) -> pd.DataFrame:
    trades = ds[ds["outcome"] != "no_fill"].sort_values("fill_time")
    hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
    idx = m5.index
    starts = idx.searchsorted(trades["fill_time"].to_numpy())

    rows = []
    for (r, s) in zip(trades.itertuples(index=False), starts):
        e = min(s + HOLD_BARS, len(m5))
        bull = r.direction == "bull"
        res = {"fill_time": r.fill_time}
        for p in policies:
            rr, why = run_trade(p, bull, r.entry, r.risk, hi[s:e], lo[s:e], cl[s:e], sp[s:e])
            res[p.name] = rr
            res[p.name + "_exit"] = why
        rows.append(res)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, policies: list[Policy]) -> list[dict]:
    out = []
    for p in policies:
        r = res[p.name]
        cum = r.cumsum()
        gw, gl = r[r > 0].sum(), -r[r < 0].sum()
        exits = res[p.name + "_exit"].value_counts(normalize=True)
        out.append({
            "policy": p.name, "label": p.label,
            "trades": int(len(r)),
            "win_pct": round(100 * float((r > 0.01).mean()), 1),
            "be_pct": round(100 * float(exits.get("be", 0)), 1),
            "avg_r": round(float(r.mean()), 3),
            "total_r": round(float(r.sum()), 1),
            "pf": round(float(gw / gl), 2) if gl else None,
            "max_dd_r": round(float((cum.cummax() - cum).max()), 1),
            "worst_year_r": round(float(r.groupby(res["fill_time"].dt.year).sum().min()), 1),
            "losing_years": int((r.groupby(res["fill_time"].dt.year).sum() < 0).sum()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Trade management comparison on Setup #1 entries")
    ap.add_argument("--symbol", default="USDJPY")
    ap.add_argument("--tf", default="M15")
    a = ap.parse_args()

    ds = load_dataset(a.symbol, a.tf)
    if ds is None:
        raise SystemExit("Zone dataset nahi mila. Pehle: python src\\research\\zone_dataset.py")
    m5 = pd.read_parquet(DATA / f"{a.symbol}_M5.parquet")

    groups = {
        "all": ds,
        "filtered": ds[(ds["disp_h1"] >= 1.5) & ds["session_fill"].isin(["London", "London+NY"])],
    }
    report = {"symbol": a.symbol, "tf": a.tf, "policies": [asdict(p) for p in POLICIES]}
    pd.set_option("display.width", 200)
    for name, g in groups.items():
        res = evaluate(g, m5, POLICIES)
        summary = summarize(res, POLICIES)
        report[name] = summary
        print(f"\n===== {name.upper()} ({len(res):,} trades) =====")
        print(pd.DataFrame(summary).drop(columns="policy").to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"{a.symbol}_{a.tf}_management.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out_file}")


if __name__ == "__main__":
    main()
