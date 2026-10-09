r"""
STEP 2 (9 Oct 2026): add H1 zones to the bot? Same walk-forward ML (unseen 2019-2026, current settings,
wick entry, tick labels, exit brain), same filters (ML top 30% + VA + activity), first touch.
  A = M15 only (CURRENT) | B = H1 only | C = M15 + H1
Trade-level stats + account simulation ($10k, 0.25%, all risk rules). Keep C only if better than A in
2019-22 AND 2023-26 (trade level) and in the account sim.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402
from src.risk import portfolio_sim as ps  # noqa: E402
pf = lambda r: round(float(r[r > 0].sum() / -r[r < 0].sum()), 3) if (r < 0).any() else np.nan  # noqa: E731


def main():
    d = wf.load(config.CORE_SYMBOLS)
    w = wf.walk_forward(d, log=lambda *a: None)
    w.to_parquet(ROOT / "data/research/step2_wf.parquet")
    base = w[(w.touch_number == 1) & w["keep_0.3"] & w.va_ok.astype(bool) & w.act_ok.astype(bool)]
    rows = []
    for name, tfs in (("A: M15 only (CURRENT)", ["M15"]), ("B: H1 only", ["H1"]), ("C: M15 + H1", ["M15", "H1"])):
        s = base[base.zone_tf.astype(str).isin(tfs)]
        r, yr = s.r_ml, s.fill_time.dt.year
        rows.append(dict(variant=name, trades=len(s), per_year=round(len(s) / 7.75), win=round(100 * float((r > 0).mean()), 1),
                         pf=pf(r), pf_2019_22=pf(r[yr <= 2022]), pf_2023_26=pf(r[yr >= 2023]), total_R=round(float(r.sum())),
                         pos_years=int((r.groupby(yr).sum() > 0).sum()),
                         pairs={p: pf(r[s.symbol.astype(str) == p]) for p in config.CORE_SYMBOLS}))
        # account sim
        sc = w.copy()
        sc["symbol"] = sc.symbol.astype(str); sc["zone_tf"] = sc.zone_tf.astype(str); sc["direction"] = sc.direction.astype(str)
        cands = []
        for tf in tfs:
            x = sc[sc.zone_tf == tf].copy()
            x["zone_tf"] = "M15"                      # load_candidates keeps M15 rows; relabel to reuse it
            cands.append(ps.load_candidates(scores=x))
        c = pd.concat(cands).sort_values("fill_time").reset_index(drop=True)
        a = ps.run(capital=10000, risk_pct=0.25, commission=5.0, cands=c, log=lambda *a, **k: None)
        rows[-1].update(acct_final=round(a["final_equity"]), acct_cagr=a["cagr_pct"], acct_maxdd=a["max_drawdown_pct"],
                        acct_pf=a["profit_factor"], acct_trades=a["trades"])
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "data/research/step2_h1.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
