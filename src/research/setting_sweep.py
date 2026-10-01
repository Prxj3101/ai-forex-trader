r"""
Settings sweep: zone ki kaunsi settings (swing size x displacement) pe POORA bot sabse achha chalta hai?

Dashboard ka MIN DISPLACEMENT slider sirf chart ke liye hai. ML ek fixed setting (swing 1, disp 2.0) pe
train hota tha. Ye script har combination ke liye POORI pipeline chalata hai:
    zone dataset v2 -> cost-aware ML (walk-forward) -> exit brain -> account simulation
aur sab ka result ek table mein deta hai. Isse data batata hai ki kaunsi setting best hai.

  swing 1 = chhota structure (3 candles), swing 2 = thoda bada (5 candles)
  displacement 1.5 = chhote moves bhi zone banayenge (zyada zones), 3.0 = sirf bade moves (kam zones)

Default: ML sirf 3 achhe pairs (EURUSD, USDJPY, GBPUSD) pe. Test mein AUD/CAD ko training mein daalne se
un 3 ka result bhi gira (account 26.7% -> 23.7%/yr, drawdown 6.7% -> 9.9%) aur AUD/CAD khud PF ~0.92 rahe.

Har combination ka data alag folder mein: data/research/sweep/s{swing}_d{disp}/ (asli files ko nahi chhedta).
Results: data/research/setting_sweep.json (har combination ek entry; dobara chalao toh update hota hai).

Chalane ka tareeka (ai-forex-trader folder se, venv active karke). Ek combination ~3-4 min:
    python src\research\setting_sweep.py --swing 1 --disp 1.5 2.0 2.5 3.0
    python src\research\setting_sweep.py --swing 2 --disp 1.5 2.0 2.5 3.0
    python src\research\setting_sweep.py --show          (sirf table dikhao)
"""

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml import walkforward  # noqa: E402
from src.research import zone_dataset_v2  # noqa: E402
from src.risk import portfolio_sim  # noqa: E402

SWEEP = ROOT / "data" / "research" / "sweep"
OUT = ROOT / "data" / "research" / "setting_sweep.json"
ALL_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "AUDUSD"]
CORE = ["EURUSD", "USDJPY", "GBPUSD"]
CAPITAL, RISK = 10_000, 0.25


def run_combo(swing: int, disp: float, symbols: list[str]) -> dict:
    folder = SWEEP / f"s{swing}_d{disp:g}"
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in symbols:
        f = folder / f"{s}_v2.parquet"
        if not f.exists():
            with redirect_stdout(io.StringIO()):
                ds = zone_dataset_v2.build(s, swing, disp, log=lambda *a, **k: None)
            ds.to_parquet(f)
        print(f"  dataset {s} ready ({time.time() - t0:.0f}s)", flush=True)

    res, wf = walkforward.train(symbols, folder=folder, save=False, log=lambda m: None)
    bot = res["bot"]
    print(f"  ML done: bot PF {bot['pf']} ({bot['trades']} trades) ({time.time() - t0:.0f}s)", flush=True)

    cands = portfolio_sim.load_candidates(scores=wf[walkforward.SCORE_COLS])
    sims = {}
    groups = [("core3", CORE)] + ([("all5", symbols)] if len(symbols) > len(CORE) else [])
    for name, syms in groups:
        r = portfolio_sim.run(CAPITAL, RISK, walkforward.COMMISSION, symbols=syms, cands=cands)
        sims[name] = {k: r[k] for k in ("cagr_pct", "max_drawdown_pct", "profit_factor", "trades_per_month",
                                        "worst_month_pct", "losing_months_pct", "yearly_return_pct", "final_equity")}
    every = next(x for x in res["summary"] if x["keep"] == 1.0)
    return dict(swing=swing, disp=disp, symbols=symbols, seconds=round(time.time() - t0),
                zones_trades=every["trades"], every_zone_pf=every["pf_trail"],
                bot=bot, sim=sims)


def load_results() -> list[dict]:
    return json.loads(OUT.read_text()) if OUT.exists() else []


def show(results: list[dict]) -> None:
    rows = []
    for r in sorted(results, key=lambda x: (len(x["symbols"]), x["swing"], x["disp"])):
        b, c3 = r["bot"], r["sim"]["core3"]
        row = dict(trained_on=len(r["symbols"]), swing=r["swing"], disp=r["disp"], bot_trades=b["trades"],
                   bot_pf=b["pf"], losing_yrs=b["losing_years"],
                   cagr=c3["cagr_pct"], max_dd=c3["max_drawdown_pct"], trades_month=c3["trades_per_month"],
                   worst_month=c3["worst_month_pct"])
        for s in ALL_PAIRS:
            row[f"pf_{s[:3]}{s[3:4]}"] = b["by_symbol"].get(s, {}).get("pf")
        rows.append(row)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Swing x displacement sweep with the full bot pipeline")
    ap.add_argument("--swing", type=int, nargs="+", default=[1])
    ap.add_argument("--disp", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--symbols", nargs="+", default=CORE,
                    help="ML kin pairs pe train ho (default 3 achhe pairs; AUD/CAD add karne se EUR/JPY/GBP bhi kharab hue)")
    ap.add_argument("--show", action="store_true", help="sirf saved results ki table")
    a = ap.parse_args()

    results = load_results()
    if not a.show:
        for sw in a.swing:
            for dp in a.disp:
                print(f"\n=== swing {sw}, displacement {dp} ===", flush=True)
                r = run_combo(sw, dp, a.symbols)
                results = [x for x in results
                           if not (x["swing"] == sw and x["disp"] == dp and x["symbols"] == a.symbols)] + [r]
                OUT.write_text(json.dumps(results, indent=2, default=str))
    show(results)


if __name__ == "__main__":
    main()
