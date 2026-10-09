"""
Bot ki central settings. Ek hi jagah: dataset, ML, dashboard, sab yahin se padhte hain.

Kyun: pehle har file mein "swing 1, displacement 2.0" alag-alag likha tha. Ab settings sweep
(src/research/setting_sweep.py, 1 Oct 2026) ke baad har pair ka apna best displacement hai.

Sweep ke results (swing 1, cost-aware ML, bot PF net of $5 commission):
    EURUSD best 1.5 (1.21),  USDJPY best 1.5 (1.28),  GBPUSD best 2.5 (1.26)
    Mixed account sim ($10k, 0.25% risk): ~45%/yr, max drawdown ~6.6%
  !! PURANE NUMBERS - fill-candle lookahead wale (M5 bars). Tick-exact (8-9 Oct 2026) ke baad
     sach: bot PF ~0.87-0.89, account sim ~-2%/yr. Edge abhi NAHI hai - details CLAUDE.md mein.
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


# ----------------------------------------------------------------------
# VALUE AREA FILTER (volume profile, 2 Oct 2026)
# ----------------------------------------------------------------------
# ML ke BAAD ek veto rule: pichle 24h ke volume profile mein entry "mehengi" ho toh trade skip.
#   r24_va_pos = entry value area (70% volume wala band) mein kahan hai, trade ki direction ke hisaab se:
#     0 = sasta kinara (buy ke liye VAL), 1 = mehenga kinara (buy ke liye VAH), >1 = VA ke bhi upar
#   r24_va_pos > VA_MAX_POS -> skip
# 0.81 = 2019 se PEHLE ke trades ka 80th percentile (test saalon se tune NAHI kiya).
# Backtest (2019-2026): bot PF 1.228 -> 1.260, max DD 8.9% -> 7.3%. Demo mein dobara verify karna hai.
# ENTRY LEVEL inside the zone (9 Oct 2026, entry_test.py on ticks): "wick" = zone edge (wick high of the demand
# candle / wick low of the supply candle) was better than "body" (body edge) in every period and pair
# (whole bot 2019-26 net PF 0.86 -> 0.91). SL stays far wick + 0.1 ATR.
ENTRY_LEVEL = "wick"


def entry_price(zone, bull: bool) -> float:
    """Limit price for a zone row (needs top/bottom/body_top/body_bottom)."""
    if ENTRY_LEVEL == "wick":
        return float(zone.top if bull else zone.bottom)
    return float(zone.body_top if bull else zone.body_bottom)


VA_FILTER = True
VA_MAX_POS = 0.81


def va_ok(va_pos):
    """True = trade allowed. Volume data na ho (NaN) toh allowed."""
    import numpy as np
    v = np.asarray(va_pos, dtype=float)
    return ~(v > VA_MAX_POS) if VA_FILTER else np.ones(v.shape, dtype=bool)


# ----------------------------------------------------------------------
# ACTIVITY FILTER (market activity at the touch, 2 Oct 2026)
# ----------------------------------------------------------------------
# ML ke BAAD dusra veto rule: zone touch ke waqt market shaant ho toh trade skip.
#   act_15m = last 15 min ka tick volume / last 24h ka average (src/features/activity.py)
#   act_15m < ACT_MIN -> skip
# 0.90 = 2021 se PEHLE ke bot trades ka 20th percentile (test saalon se tune NAHI kiya).
# Asli CME orderflow test se mila: sirf "market active hai ya nahi" kaam karta hai, aur free tick volume se
# utna hi achha naapa jaata hai. Backtest: PF 1.22 -> 1.30, max DD 7.5% -> 6.2%. Demo mein verify karna hai.
ACT_FILTER = True
ACT_MIN = 0.90


def act_ok(act):
    """True = trade allowed. Volume data na ho (NaN) toh allowed."""
    import numpy as np
    v = np.asarray(act, dtype=float)
    return ~(v < ACT_MIN) if ACT_FILTER else np.ones(v.shape, dtype=bool)


# ----------------------------------------------------------------------
# NEWS BLOCK (har account pe, 2 Oct 2026)
# ----------------------------------------------------------------------
# High-impact news (Forex Factory "High") ke NEWS_BLOCK_BEFORE_MIN min pehle se NEWS_BLOCK_AFTER_MIN min baad
# tak us news ki currency wale pair pe NAYA trade / naya limit order nahi. Khule trades ko kuch nahi hota.
# Kyun: news pe spread / slippage backtest se zyada hota hai. Test (2019-2025, 1-2 pip news slippage):
#   bina block 21-45%/yr (DD 15% tak), -5/+20 block 26-46%/yr (DD ~6-8.6%). -5/+30 thoda safer, -5/+5 thoda kam.
# Demo mein asli slippage naap ke window chhota/bada karenge.
NEWS_BLOCK = True
NEWS_BLOCK_BEFORE_MIN = 5
NEWS_BLOCK_AFTER_MIN = 20


# ----------------------------------------------------------------------
# LIVE: SPREAD + COMMISSION (3 Oct 2026)
# ----------------------------------------------------------------------
# ML ne Dukascopy spread (~0.3-0.7 pip) + $5 commission pe seekha. Live broker ka spread alag hota hai
# (MetaQuotes demo / Funding Pips JPY ~0.0-0.1). Isliye do kaam alag:
#   ML_SPREAD_FLOOR = True -> ML ko spread = max(live, Dukascopy normal) dikhao (src/live/spread_floor.py)
#                             => trade lena / skip karna backtest jaisa
#   BROKER_COMMISSION      -> shadow P&L ke liye ASLI account ka commission (per lot, round trip)
#                             Funding Pips $5, The5ers $4, Exness Raw / Zero $5 (EUR/JPY; GBP Zero $6.25)
# "Standard" type account (commission 0, spread ~1 pip) bot ke liye mat lena.
ML_SPREAD_FLOOR = True
BROKER_COMMISSION = 5.0

# Broker ki MT5 ghadi (server time) UTC se kitni aage hai:
#   None   = khud naapo (market khula: live tick se; weekend: Friday ke close se)
#   "ny+7" = New York + 7h (MetaQuotes demo, kai brokers; DST apne aap)
#   0      = UTC (Exness), 2 / 3 = fixed ghante
BROKER_UTC_OFFSET = None


# ----------------------------------------------------------------------
# ACCOUNT PROFILES: kis type ke account pe bot chal raha hai -> kaunse extra rules
# ----------------------------------------------------------------------
# Funding Pips MASTER (funded) news rule (screenshot, 1 Oct 2026):
#   - high-impact NEWS: trade 5 min pehle se 5 min baad tak OPEN ya CLOSE hua -> profit kat jaata hai
#   - SPEECH: 10 min pehle se speech KHATAM hone ke 10 min baad tak
#   - exception: trade event se >= 5 ghante pehle khula tha -> window mein close ho sakta hai
#   - partial close = poora order flag (bot kabhi partial close nahi karta)
# Evaluation phase aur real broker: koi news rule nahi.
#
# Hum firm ke 5/10 min ki jagah thoda zyada margin rakhte hain (firm ka calendar alag hai,
# time 1-2 min idhar udhar ho sakta hai).
#
#   news_rules       : rules on/off
#   news_before_min  : news se itne min pehle se window shuru
#   news_after_min   : news ke itne min baad tak window
#   speech_before_min / speech_after_min : speech ke liye (after = speech khatam hone ke baad)
#   exempt_hours     : trade event se itne ghante pehle khula ho toh window mein close allowed
#   pre_news         : "close" = window se pehle trade band kar do (backtest mein best)
#                      "hold"  = trade chalne do; window mein exit hua toh profit kat jayega
#   close_buffer_min : window shuru hone se kitne min pehle close kar dena
#   impacts          : calendar ke kaunse impact levels gine jaayein
ACCOUNT_PROFILES = {
    "prop_master": dict(
        label="PROP FUNDED (NEWS RULES)", news_rules=True,
        news_before_min=10, news_after_min=10,
        speech_before_min=15, speech_after_min=15,
        exempt_hours=5.0, pre_news="close", close_buffer_min=2,
        impacts=("High",),
    ),
    "prop_evaluation": dict(label="PROP EVALUATION (NO NEWS RULES)", news_rules=False),
    "real_broker": dict(label="REAL BROKER (NO NEWS RULES)", news_rules=False),
}

# Abhi kaunsa account? Funding Pips abhi EVALUATION phase mein hai (1 Oct 2026) -> news rules off.
# Funded (Master) milte hi yahan "prop_master" kar dena.
ACCOUNT_PROFILE = "prop_evaluation"


def profile(name: str | None = None) -> dict:
    name = name or ACCOUNT_PROFILE
    p = dict(ACCOUNT_PROFILES[name])
    p["name"] = name
    return p
