r"""
OUT-OF-SAMPLE CHECK of the H1 supply & demand rules on pairs NOT used in the first test (AUDUSD, USDCAD).
Same rules as sd_test.py (SD_TF=H1). No tick data for these pairs, so entries/exits run on M5 candles:
  - market entries (normal / conservative) = open of the next M5 bar at ask (BUY) / bid (SELL)  -> no fill-candle issue
  - limit fills (aggressive): in the fill bar only the stop counts and TP only if the bar CLOSE is beyond it
    (price before the fill is never counted); later bars: stop first, then target (conservative)
Chalana:  python src\research\sd_validate.py AUDUSD 2019   |   python src\research\sd_validate.py report
"""
import os, sys
os.environ["SD_TF"] = "H1"
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import sd_test as sd  # noqa: E402

sd.PIP.update({"AUDUSD": 1e-4, "USDCAD": 1e-4})
sd.OUT = ROOT / "data/research/sd_test_H1_validate"
sd.OUT.mkdir(parents=True, exist_ok=True)
HOLD = 3 * 288


class M5Engine:
    def __init__(self):
        self.cache, self.limit_fills = {}, set()

    def m5(self, sym):
        if sym not in self.cache:
            m = pd.read_parquet(ROOT / f"data/processed/{sym}_M5.parquet", columns=["open", "high", "low", "close", "spread"])
            self.cache[sym] = (m.index.as_unit("ns").asi8 // 1_000_000, *(m[k].to_numpy() for k in ("open", "high", "low", "close", "spread")))
        return self.cache[sym]

    def ticks(self, sym, t0, t1):
        t, o, h, l, c, sp = self.m5(sym)
        i = int(np.searchsorted(t, t0))
        if i >= len(t) or t[i] >= t1:
            return np.array([]), np.array([]), np.array([])
        return np.array([t[i]]), np.array([o[i] + sp[i]]), np.array([o[i]])

    def find_fill(self, sym, bull, entry, s, e):
        t, o, h, l, c, sp = self.m5(sym)
        i, j = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        hit = (l[i:j] + sp[i:j] <= entry) if bull else (h[i:j] >= entry)
        if not hit.any():
            return None
        f = int(t[i + int(np.argmax(hit))])
        self.limit_fills.add((sym, f))
        return f

    def run_exit(self, sym, bull, entry, sl, fill_ms, policy, tp_r=None):
        t, o, h, l, c, sp = self.m5(sym)
        tp = {"tp1": 1.0, "tp2": 2.0}.get(policy, tp_r)
        risk = abs(entry - sl)
        i0 = int(np.searchsorted(t, fill_ms)); i1 = min(i0 + HOLD, len(t))
        limit = (sym, fill_ms) in self.limit_fills
        for i in range(i0, i1):
            fav = ((h[i] - entry) if bull else (entry - (l[i] + sp[i]))) / risk
            adv = ((l[i] - entry) if bull else (entry - (h[i] + sp[i]))) / risk
            if adv <= -1:
                return dict(r=-1.0, reason="sl", exit_ms=int(t[i]), best_r=np.nan)
            if i == i0 and limit:
                fav = ((c[i] - entry) if bull else (entry - (c[i] + sp[i]))) / risk
            if fav >= tp:
                return dict(r=tp, reason="tp", exit_ms=int(t[i]), best_r=tp)
        x = c[i1 - 1] if bull else c[i1 - 1] + sp[i1 - 1]
        return dict(r=((x - entry) if bull else (entry - x)) / risk, reason="timeout", exit_ms=int(t[i1 - 1]), best_r=np.nan)


sd.te = M5Engine()

if __name__ == "__main__":
    if sys.argv[1] == "report":
        sd.report()
    else:
        sd.run(sys.argv[1], int(sys.argv[2]))
