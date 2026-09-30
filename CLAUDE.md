# Forex AI Trading Agent - Project Notes

Handoff notes from an earlier chat (inside the QubeEx project) where this idea was discussed.
Read this first before doing anything in this project.

## About the user
- Name: Prajwal. Talks in casual Hinglish (Hindi + English). Reply the same way.
- Computer Engineering graduate (University of Mumbai). Knows Python, JavaScript, TypeScript, React, React Native/Expo, PostgreSQL, REST APIs.
- Built QubeEx (a full-stack AI assessment platform) - that is a separate project, not related to this one.
- Working style: one step at a time. Do not start building or writing code until he clearly says go ahead. If he says "ruk" (wait), stop and wait.
- He wants to LEARN while building. For every step, command, tool or piece of code: explain what it is, why we need it, and what it does, in simple Hinglish. Assume he does not know terms like venv, pip, etc. unless he says so.

## Goal
- Build an AI trading agent that trades forex on its own: decides when to enter, when to exit, and how much to risk.
- Pairs: XAUUSD (gold), EURUSD, GBPUSD, USDJPY, USDCAD, AUDUSD.
- Long-term idea: give it a small real account (about 100 USD) and grow it.
- Agreed approach: backtesting first, then paper trading on a demo account for 2-3 months. Real money only if results are consistently positive after fees, and only an amount he can afford to lose.

## Things already agreed / said honestly
- Building the system is fully doable. Profitability is NOT guaranteed and nobody can promise it. He said he is comfortable with that and wants to at least build the system.
- The agent is a system of parts, not one magic AI. An LLM should not make the live entry decisions (too slow, not reliable on price). Use it for news/sentiment summaries and trade journaling.
- Legal note (India): trading these non-INR pairs with real money through an offshore broker (Exness, OANDA, MT5 brokers etc.) goes against FEMA; RBI keeps an Alert List of unauthorised forex platforms. Backtesting and demo/paper trading are fine since no money is sent abroad. Real-money decision comes later and is his call.

## Planned architecture
1. Data: historical candles for backtesting; live prices from a demo account; economic calendar (NFP, FOMC, CPI).
2. Market context: trend vs range, volatility (ATR), session (London / New York / Asia), no trading right before big news.
3. Strategy: rule-based base (trend-following / breakout / mean reversion) + later an ML model as a filter (probability the setup works), not raw price prediction. Exits: stop-loss, take-profit, trailing stop, time-based exit.
4. Risk manager (most important): 0.5-1% risk per trade, ATR-based stops, max daily/weekly loss (bot stops itself), correlation control (all pairs contain USD, so 5 open trades = one big USD bet), separate sizing for XAUUSD (much more volatile).
5. LLM role: news/sentiment summary, trade journal (why entered, why exited, what went wrong).
6. Testing: include spread, slippage, swap in backtests; walk-forward testing to avoid overfitting; judge on profit factor, max drawdown, number of trades (not just win rate).
7. Later: dashboard to see trades and performance; runs on his PC or a VPS.

## Phase plan
1. Data + backtesting setup (Python; backtesting.py or vectorbt).
2. One simple strategy first (e.g. trend-following on EURUSD) with the risk manager.
3. Test and improve, then add other pairs and the ML filter.
4. Paper trade live on a demo account for 2-3 months.

## Open decisions (not answered yet)
- Platform for demo trading + live data: MetaTrader 5 demo (recommended - free, Windows, Python `MetaTrader5` package, has gold and all pairs) vs OANDA practice API.
- (Resolved) Laptop has Python 3.14.3, pip 25.3, Git 2.53. Checked: MetaTrader5, numba, LightGBM, vectorbt, backtesting.py, TA-Lib all support 3.14.

## Decisions made (Sept 2026 chat)
- Vision: the agent is its own "market brain" (rules + self-trained ML model like LightGBM, retrained over time), not an LLM. LLM only for news/journal.
- His trading style is MULTI-TIMEFRAME (top-down): Weekly, Daily, 4H, 1H for bias/structure, execute on 15m or 5m. Agent must work the same way. Plan: download M5 data, resample to all higher TFs ourselves; always use only CLOSED higher-TF candles (no lookahead).
- Trade in all 3 sessions: Asian (Tokyo), London, New York. Sessions defined in local time with DST via tz names (Asia/Tokyo, Europe/London, America/New_York), never fixed UTC hours. Session is a feature for the ML model; behaviour differs per session.
- All data stored in UTC. Daily/weekly candles close at NY 5 PM (convert to America/New_York, handles DST automatically).
- STRATEGY ARCHITECTURE (his idea): one file per strategy in src/strategies/ (e.g. EMA+RSI, ICT, London breakout, and "his combo" = everything he uses on TradingView for gold). All strategies share one interface, the same features, backtester and risk manager, so they can be swapped and compared. He does NOT want basic EMA as the main strategy (can stay as a baseline/benchmark only). On TradingView he uses ICT Sessions & Killzones + CPR (details to come from him).
- ORDER FLOW: he means both (a) SMC order flow (price-action institutional direction) and (b) real order flow (delta/footprint). His current charting is SMC order flow. Plan: Phase 1 SMC brain on free data; Phase 2 add real order flow from CME/COMEX futures trades data (GC gold; 6E/6B/6J/6C/6A for FX) using Databento $125 free credits for historical; Phase 3 live data only if it adds edge (~$12-40/month).
- SETUP #1 "Order Block Tap" (from his USDJPY trade, 30 Sep 2026): strong displacement that breaks structure -> zone = last opposite candle before displacement (bullish: last red candle; bearish: last green candle), on whichever TF (15m/1H/4H) it is cleanest -> LIMIT order at zone (no confirmation) -> SL beyond zone with a small buffer -> TP min 1:2, let it run to next liquidity (swing high/low). Killzone (e.g. London open).
  Things he explicitly wants the bot/ML to FIND FROM DATA (do not hardcode): zone with wick vs body only; whether a zone works once or multiple touches; invalidation (close through zone vs price sitting inside zone); SL buffer size. "Strong displacement" threshold to be tuned (start ~1.5-2x ATR body + breaks swing).
- (Older plan) First model: XAUUSD, standard trend-following pullback strategy (originally planned on 1H; now to be adapted to multi-timeframe) (EMA 50/200 trend, pullback to EMA 20/50, ATR stop, 1:2 RR / trailing). His own manual strategies come later.
- Zero cost at start: Groq free API instead of Claude Haiku for news; laptop instead of VPS (Azure free Windows VM later for 24x5 paper trading). Pay only if it works.
- Jev (TypeSafe AI classifier, launched Sept 2026): not for trade decisions; maybe test later for the news layer.
- He uses Antigravity (VS Code based) with CMD terminal: activate venv with `.venv\Scripts\activate.bat`.

## Progress
- Step 1 done: venv (.venv), installed pandas, numpy, pyarrow, backtesting, ta, matplotlib, jupyter; requirements.txt; folders data/raw, data/processed, notebooks, src/{data,strategies,risk,backtest}, reports; .gitignore; git init + first commit.
- Step 2 done: src/data/download_dukascopy.py (dukascopy-python, UTC-aware dates, BID+ASK, per-year Parquet files, resume/skip). XAUUSD H1 2016-2026 downloaded to data/raw (22 files).
- Folder renamed forex-agent -> C:\Users\Prxj\ai-forex-trader; venv recreated from requirements.txt.
- Pushed to private GitHub repo https://github.com/Prxj3101/ai-forex-trader (branch main). Claude must NOT run git commands on his repo from the sandbox (left a stale .git/index.lock once).
- Step 3 done: download script takes --tf (M1/M5/M15/H1). XAUUSD M5 2016 -> 2026-09-25 downloaded.
- Step 4 done: src/data/build_timeframes.py -> data/processed/XAUUSD_{M5,M15,H1,H4,D1,W1}.parquet. Columns: open, high, low, close (BID), volume, spread (ask-bid close), close_time (UTC, for no-lookahead), m5_count. Index `time` = candle open (UTC). 760,923 M5 candles; only 3 flat candles, 0 bad prices; 74 non-weekend gaps of ~5h = US holidays (CME early close), expected, not missing data.
- Step 5: notebooks/01_explore_data.ipynb (he runs it via `jupyter lab`). Findings (last 2 yrs, M5): avg spread ~$0.6; spread/range: London+NY overlap best 10.4%, London 15.6%, Asian 16.1%, NY-only 17.2%, Off-hours worst 21.5%. Most volatile IST hours 18-20 (overlap + US data), then 6-7 IST (Asian open); quietest 2 AM and 9-10 AM IST. D1 candles = 276 M5 (23h, 1h daily break). Gold went $1,074 (2016) -> ~$4,300 (Sept 2026).
- D1/W1 now have `trade_date` column (TradingView/MT5 date; D1 `time` is previous evening 6 PM NY). Verified vs TradingView OANDA:XAUUSD 22 Sep 2026: H/L within $0.5, close within $2, open ~$5 (broker first-tick difference). Data verified OK.
- His manual trade export: data/manual_trades/profitable_trades.csv (order fills, XAU/USD, Mar-Sep 2026, no P&L column, no position IDs). FIFO-reconstructed 272 round trips. Entry prices 100% inside matching Dukascopy M5 candles (same feed level). Findings: scalps <=1h (139, median hold 8 min): 95.7% win, avg +6.5 pts, PF ~14; intraday 1-24h (71): 56% win, PF 2.5; swings >1 day (62, mostly BUY): 48% win, PF 0.85, net loser. Top 10 losses = 54% of all losses (held days/weeks without stop). stop_loss exits median +0.23 pts (moves SL to breakeven). More SELLs than BUYs in scalps (94 vs 45). Most entries 14:00-21:00 IST. HTF trend / CPR alignment showed no clear edge for scalps. Not enough to train ML directly (small, one regime, mostly winners, no entry reasons); use as validation set for his combo strategy + risk rules (max hold, mandatory SL). Scalps with ~8 min holds need M1 data for accurate backtests.
- notebooks/02_my_trades_review.ipynb exists (charts of his CSV trades) but he found it hard to read; he prefers sending his own TradingView screenshots + template explanations instead.
- download_dukascopy.py and build_timeframes.py now take --symbol (one or more of XAUUSD, EURUSD, GBPUSD, USDJPY, USDCAD, AUDUSD).
- src/features/smc.py: atr(), find_swings(), find_orderflow_zones() (he calls OBs "orderflow zones"; named to match), track_zones(). notebooks/03_orderflow_zones_check.ipynb plots zones in NY time.
- USDJPY M5 downloaded and processed (803,684 M5 candles, 0 bad/flat/duplicates; only gaps = Christmas/New Year closures). Validation on his 29-30 Sep trade (M15): with swing_n=1, min_disp=2.0 the bot finds his demand zone exactly: zone candle 29 Sep 23:15 NY, wick 156.551-156.670, body top 156.651 = his exact entry price; first touch 30 Sep 03:15 NY (London open) = his entry. Needs swing_n=1 (internal/minor structure); swing_n>=2 misses it because the waterfall drop had no fractal swing highs. Supply ~157.39-157.46 found on M15 (29 Sep 19:45) and H1 (29 Sep 19:00, 157.335-157.456, only with min_disp 1.5). Insight: he enters at zone BODY top with limit, SL below WICK low + ~1-2 pip buffer.
- DASHBOARD: he wants everything visible in one place, live, as a sci-fi "command center" (dark, neon, agents visibly working in real time, immersive but easy to understand). Plan: FastAPI (Python backend) + WebSocket live events + React/Vite/TS/Tailwind frontend + TradingView Lightweight Charts + Framer Motion. "Agents" = system layers (Data, Structure, Zone Hunter, ML Brain, Risk, Execution). Rule: dashboard only displays/controls; trading logic stays in Python. Node v20.20.1 installed. Static look-and-feel mockup: dashboard/mockup/command_center_mockup.html (fake data), v0.3 approved in principle: NASA-style DESIGN (boot sequence, starfield, CRT scanlines, angled panels, neural-net animation, session radar, ticker, Orbitron + Share Tech Mono fonts) but ALL WORDING MUST BE TRADING TERMS, no NASA/mission words, and ENGLISH ONLY in the dashboard (no Hinglish, including tooltips). Chat with him stays Hinglish.
- Dashboard step 1 (backend) done: src/api/events.py (EventBus, thread-safe emit -> WebSocket broadcast, keeps last 200 events) and src/api/server.py (FastAPI: /api/health, /api/symbols, /api/candles, /api/zones, POST /api/scan runs a 10-year zone scan in a background thread with live agent events, WS /ws). Run: `uvicorn src.api.server:app --reload`, docs at localhost:8000/docs. Tested with TestClient. First real scan: USDJPY M15 swing_n=1 disp=2.0 -> 11,873 zones, 91.7% revisited, 82.2% later invalidated, avg 2.17 touches, 42% with FVG (0.8 s).
- Next: dashboard step 2 = React frontend scaffold in dashboard/frontend (Vite + React + TS + Tailwind + lightweight-charts + framer-motion), port 5173.
- Pending after dashboard: zone dataset (result cards per zone + features incl. HTF displacement, post-BOS move size, multi-TF confluence) because he rejected some detected zones as "no real move" / "too small move" and wants the bot to learn worthiness from history.
- (old) Next: download USDJPY M5 + build timeframes; then code OB detector v1 and check it finds his two USDJPY zones of 29-30 Sep 2026 (demand ~156.55-156.72, 15m; supply ~157.33-157.44, 1H).

## Other idea (separate, for later)
- He also wants to discuss an AI-based mental health business idea. Not started.
