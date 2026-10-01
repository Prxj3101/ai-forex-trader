"""
Bot ki central settings. Ek hi jagah: dataset, ML, dashboard, sab yahin se padhte hain.

Kyun: pehle har file mein "swing 1, displacement 2.0" alag-alag likha tha. Ab settings sweep
(src/research/setting_sweep.py, 1 Oct 2026) ke baad har pair ka apna best displacement hai.

Sweep ke results (swing 1, cost-aware ML, bot PF net of $5 commission):
    EURUSD best 1.5 (1.21),  USDJPY best 1.5 (1.28),  GBPUSD best 2.5 (1.26)
    Mixed account sim ($10k, 0.25% risk): ~45%/yr, max drawdown ~6.6%
GBPUSD ka spread zyada hai, isliye usko sirf strong (2.5x) zones chahiye.
"""

SWING_N = 1                 # swing 1 = chhota structure; sweep mein swing 2 se har baar behtar

# Bot sirf in pairs pe trade karta hai (AUD/CAD pe Setup #1 kharche ke baad loss mein hai)
CORE_SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD"]

# Har pair ka minimum displacement (ATR mein) jisse zone banta hai
DISP_BY_SYMBOL = {
    "EURUSD": 1.5,
    "USDJPY": 1.5,
    "GBPUSD": 2.5,
}
DEFAULT_DISP = 2.0          # jo pair list mein nahi hai (research ke liye)


def disp_for(symbol: str) -> float:
    return DISP_BY_SYMBOL.get(symbol, DEFAULT_DISP)
