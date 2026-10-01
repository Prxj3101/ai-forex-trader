r"""
Zone settings grid: zone mein ENTRY kahan lagaye aur SL buffer kitna rakhe? (data decide kare)

Prajwal ne kaha tha ye cheezein hardcode mat karo, data se nikalo:
  - Entry: wick edge / body edge (abhi wala) / zone ka middle / body ka door wala edge
  - SL buffer: zone ki wick ke paar kitna extra (ATR mein)

Har combination ko 10 saal ke M15 zones (pehla touch) pe chalate hain, 2 exits ke saath (1:1 aur trailing),
aur result COMMISSION + SLIPPAGE ke baad (net R) dekhte hain. Kyunki entry/SL badalne se SL ka size
badalta hai, aur chhote SL pe commission zyada R khata hai.

Overfitting check: 2016-2021 (pehla half) aur 2022-2026 (doosra half) alag-alag bhi dikhate hain.
Jo setting DONO halves mein achhi ho, wahi bharose layak.

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\research\zone_settings.py --symbol EURUSD USDJPY GBPUSD
Output: terminal table + data/research/zone_settings.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.smc import find_orderflow_zones  # noqa: E402
from src.research.management import POLICIES, run_trade_full  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "research"
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}
POL = {p.name: p for p in POLICIES}
WAIT = 3 * 288
HOLD = 3 * 288

ENTRIES = ["wick_edge", "body_edge", "zone_mid", "body_far"]
BUFFERS = [0.05, 0.1, 0.2, 0.3, 0.5]


def entry_price(mode: str, bull: bool, top, bottom, body_top, body_bottom) -> float:
    """Bull zone (demand) ke liye; bear mein sab ulta (mirror)."""
    if mode == "wick_edge":          # zone ka pehla kinara: sabse pehle fill, sabse bada SL
        return top if bull else bottom
    if mode == "body_edge":          # abhi wala (Prajwal ki entry)
        return body_top if bull else body_bottom
    if mode == "zone_mid":           # poore zone (wick se wick) ka beech
        return (top + bottom) / 2
    if mode == "body_far":           # body ka door wala kinara: gehri entry, chhota SL, kam fills
        return body_bottom if bull else body_top
    raise ValueError(mode)


def run_symbol(sym: str, commission: float, slip_pips: float) -> pd.DataFrame:
    m15 = pd.read_parquet(DATA / f"{sym}_M15.parquet")
    m5 = pd.read_parquet(DATA / f"{sym}_M5.parquet", columns=["high", "low", "close", "spread"])
    hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
    idx = m5.index
    z = find_orderflow_zones(m15, swing_n=1, min_disp_atr=2.0)
    usd_quote = sym.endswith("USD")
    rows = []
    for r in z.itertuples(index=False):
        bull = r.direction == "bull"
        s = int(idx.searchsorted(r.created_at))
        e = min(s + WAIT, len(m5))
        if s >= len(m5):
            continue
        for mode in ENTRIES:
            entry = entry_price(mode, bull, r.top, r.bottom, r.body_top, r.body_bottom)
            hit = np.nonzero((lo[s:e] + sp[s:e] <= entry) if bull else (hi[s:e] >= entry))[0]
            if hit.size == 0:
                continue
            f = s + int(hit[0])
            g = min(f + HOLD, len(m5))
            for buf in BUFFERS:
                sl = (r.bottom - buf * r.atr) if bull else (r.top + buf * r.atr)
                risk = abs(entry - sl)
                if risk <= 0:
                    continue
                per_lot = risk * 100_000 if usd_quote else risk * 100_000 / entry     # SL pe 1 lot ka USD nuksaan
                cost_r = commission / per_lot                                          # commission R mein
                slip_r = slip_pips * PIP[sym] / risk
                row = dict(symbol=sym, entry_mode=mode, buffer=buf, fill_time=idx[f],
                           sl_pips=risk / PIP[sym])
                for name in ("tp1", "trail1"):
                    rr, why, _ = run_trade_full(POL[name], bull, entry, risk, hi[f:g], lo[f:g], cl[f:g], sp[f:g])
                    row[f"gross_{name}"] = rr
                    row[f"net_{name}"] = rr - cost_r - (slip_r if why in ("sl", "trail", "be") else 0.0)
                rows.append(row)
    return pd.DataFrame(rows)


def pf(r: pd.Series) -> float:
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl else float("nan")


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (mode, buf), g in d.groupby(["entry_mode", "buffer"]):
        early = g[g["fill_time"].dt.year <= 2021]
        late = g[g["fill_time"].dt.year >= 2022]
        row = dict(entry=mode, buffer=buf, trades=len(g), median_sl_pips=round(float(g["sl_pips"].median()), 1))
        for name, lab in (("tp1", "1:1"), ("trail1", "trail")):
            col = f"net_{name}"
            yearly = g.groupby(g["fill_time"].dt.year)[col].sum()
            row.update({
                f"gross_pf_{lab}": pf(g[f"gross_{name}"]),
                f"net_pf_{lab}": pf(g[col]),
                f"net_R_{lab}": round(float(g[col].sum()), 0),
                f"net_pf_{lab}_16_21": pf(early[col]),
                f"net_pf_{lab}_22_26": pf(late[col]),
                f"losing_years_{lab}": int((yearly < 0).sum()),
            })
        out.append(row)
    return pd.DataFrame(out).sort_values("net_R_1:1", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zone entry / SL buffer grid (net of costs)")
    ap.add_argument("--symbol", nargs="+", default=["EURUSD", "USDJPY", "GBPUSD"])
    ap.add_argument("--commission", type=float, default=7.0, help="USD per 1.00 lot round trip")
    ap.add_argument("--slip", type=float, default=0.2, help="pips of slippage on stop exits")
    a = ap.parse_args()

    pd.set_option("display.width", 250)
    parts, report = [], {"commission": a.commission, "slippage_pips": a.slip, "by_symbol": {}}
    for sym in a.symbol:
        t0 = time.time()
        d = run_symbol(sym, a.commission, a.slip)
        parts.append(d)
        s = summarize(d)
        report["by_symbol"][sym] = s.to_dict("records")
        print(f"\n===== {sym} ({time.time() - t0:.0f}s) =====")
        print(s.to_string(index=False))
    s = summarize(pd.concat(parts))
    report["all"] = s.to_dict("records")
    print("\n===== ALL PAIRS =====")
    print(s.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zone_settings.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nsaved -> data/research/zone_settings.json")


if __name__ == "__main__":
    main()
