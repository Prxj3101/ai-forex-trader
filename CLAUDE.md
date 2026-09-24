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
- First model: XAUUSD, standard trend-following pullback strategy (originally planned on 1H; now to be adapted to multi-timeframe) (EMA 50/200 trend, pullback to EMA 20/50, ATR stop, 1:2 RR / trailing). His own manual strategies come later.
- Zero cost at start: Groq free API instead of Claude Haiku for news; laptop instead of VPS (Azure free Windows VM later for 24x5 paper trading). Pay only if it works.
- Jev (TypeSafe AI classifier, launched Sept 2026): not for trade decisions; maybe test later for the news layer.
- He uses Antigravity (VS Code based) with CMD terminal: activate venv with `.venv\Scripts\activate.bat`.

## Progress
- Step 1 done: venv (.venv), installed pandas, numpy, pyarrow, backtesting, ta, matplotlib, jupyter; requirements.txt; folders data/raw, data/processed, notebooks, src/{data,strategies,risk,backtest}, reports; .gitignore; git init + first commit.
- Step 2 done: src/data/download_dukascopy.py (dukascopy-python, UTC-aware dates, BID+ASK, per-year Parquet files, resume/skip). XAUUSD H1 2016-2026 downloaded to data/raw (22 files).
- Folder renamed forex-agent -> C:\Users\Prxj\ai-forex-trader; venv recreated from requirements.txt.
- Pushed to private GitHub repo https://github.com/Prxj3101/ai-forex-trader (branch main). Claude must NOT run git commands on his repo from the sandbox (left a stale .git/index.lock once).
- Next: download M5 data and resample to 15m/1H/4H/D/W (daily close at NY 5 PM).

## Other idea (separate, for later)
- He also wants to discuss an AI-based mental health business idea. Not started.
