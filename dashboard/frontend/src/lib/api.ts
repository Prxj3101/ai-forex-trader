// Backend (FastAPI) se baat karne wala code. Saare types yahan hain.

export const API = 'http://localhost:8000'
export const WS_URL = 'ws://localhost:8000/ws'

export type Candle = {
  time: number
  open: number
  high: number
  low: number
  close: number
  spread: number
}

export type Zone = {
  direction: 'bull' | 'bear'
  zone_candle_time: number
  created_at: number
  top: number
  bottom: number
  body_top: number
  body_bottom: number
  entry?: number | null          // bot's limit price (wick edge since 9 Oct 2026); missing -> wick edge
  disp_atr: number
  has_fvg: boolean
  first_touch_time: number | null
  first_touch_depth: number | null
  invalidated_time: number | null
  touches: number
}

export type AgentId = 'data' | 'structure' | 'zone' | 'ml' | 'risk' | 'execution' | 'system'

export type AgentEvent = {
  ts: number
  agent: AgentId
  message: string
  status: 'active' | 'idle' | 'done' | 'error'
  progress: number | null
  kind: 'log' | 'result' | 'research' | 'ml'
  data: Record<string, unknown>
}

export type ScanResult = {
  symbol: string
  tf: string
  swing_n: number
  min_disp: number
  zones: number
  bull: number
  bear: number
  touched_pct: number
  invalidated_pct: number
  avg_touches: number
  with_fvg_pct: number
  seconds: number
}

export type Stats = {
  zones: number
  trades: number
  win_pct: number | null
  avg_r: number | null
  total_r: number
  pf: number | null
  max_dd_r?: number
}

export type BreakdownRow = Stats & { label: string }

export type ResearchReport = {
  symbol: string
  tf: string
  settings: { swing_n: number; min_disp: number; rr: number }
  break_even_win_pct: number
  all: Stats
  filtered: Stats
  equity_all: { time: number; value: number }[]
  equity_filtered: { time: number; value: number }[]
  by_session: BreakdownRow[]
  by_disp: BreakdownRow[]
  by_year: BreakdownRow[]
  by_direction: BreakdownRow[]
}

export type ResearchFilters = {
  min_disp_h1: number
  sessions: string[]
  fvg_only: boolean
  confluence_only: boolean
  direction: 'both' | 'bull' | 'bear'
}

export type MLSummaryRow = {
  keep: number
  trades: number
  win_pct: number
  pf_trail: number
  pf_1to1: number
  pf_1to2: number
  total_r: number
  avg_r: number
  losing_years: number
}

export type MLReport = {
  trained_at: string | null
  symbols: string[]
  n_trades: number
  n_features: number
  first_test_year: number
  summary: MLSummaryRow[]
  score_buckets: { bucket: string; trades: number; avg_score: number; real_win_pct: number; pf: number }[]
  by_year: { keep: number; year: number; trades: number; pf: number; total_r: number }[]
  by_symbol: { keep: number; symbol: string; trades: number; pf: number; total_r: number }[]
  importance: { feature: string; pct: number }[]
  exits: { exit: string; label: string; trades: number; pf: number; total_r: number; max_dd_r: number; losing_years: number }[]
  bot: {
    label: string
    trades: number
    win_pct: number
    pf: number
    total_r: number
    avg_r: number
    max_dd_r: number
    losing_years: number
    trail_share_pct: number
    by_year: Record<string, number>
    by_symbol: Record<string, { trades: number; pf: number; total_r: number }>
  } | null
  commission: number | null
  slippage_pips: number | null
  net: boolean
  va_filter?: { on: boolean; max_pos: number; removed: number; pf_without: number; trades_without: number } | null
  act_filter?: { on: boolean; min_act: number; removed: number; pf_without: number; trades_without: number } | null
}

export type RiskReport = {
  combo?: { aggressive_trades: number; conservative_trades: number }
  config: {
    capital: number
    risk_pct: number
    commission_per_lot: number
    max_open: number
    daily_loss_pct: number
    slippage_pips: number
    brake_dd_pct?: number
    brake_factor?: number
    kill_dd_pct?: number
    daily_risk_budget_pct?: number
    house_money_pct?: number
    risk_cap_pct?: number
    combo_enabled?: boolean
    combo_reduce_dd_pct?: number
    combo_recover_dd_pct?: number
  }
  braked_trades_pct?: number
  worst_day_pct?: number | null
  risk_used?: { avg_pct: number; max_pct: number; boosted_trades_pct: number } | null
  kill_switch_events?: { time: string; drawdown_pct: number }[]
  va_filter?: { on: boolean; max_pos: number; skipped: number }
  act_filter?: { on: boolean; min_act: number; skipped: number }
  account_profile?: {
    name: string
    label: string
    news_rules: boolean
    news_block?: boolean
    block_before_min?: number
    block_after_min?: number
    blocked_fills?: number
    closed_before_news?: number
    profit_removed?: number
    calendar_end?: string | null
    news_before_min?: number
    news_after_min?: number
    speech_before_min?: number
    speech_after_min?: number
    exempt_hours?: number
    pre_news?: 'close' | 'hold'
  }
  start: string
  end: string
  years: number
  trades: number
  trades_per_month: number
  start_equity: number
  final_equity: number
  total_return_pct: number
  cagr_pct: number
  max_drawdown_pct: number
  win_pct: number
  profit_factor: number
  commission_usd: number
  losing_months_pct: number
  worst_month_pct: number
  best_month_pct: number
  longest_losing_streak: number
  skipped: Record<string, number>
  yearly_return_pct: Record<string, number>
  by_symbol: Record<string, { trades: number; pnl: number }>
  equity_curve: { time: number; value: number }[]
}

export type MLScore = {
  fill_time: number
  zone_candle_time: number
  direction: 'bull' | 'bear'
  touch_number: number
  entry: number
  sl: number
  score: number
  keep_top30: boolean
  keep_top50: boolean
  r_trail: number
  exit: string
  exit_ml: 'tp1' | 'trail' | null
  r_ml: number | null
  va_ok?: boolean
  va_pos?: number | null
  act_ok?: boolean
  act?: number | null
}

export type ResearchExit = 'tp1' | 'tp2' | 'trail' | 'brain'

export type ResearchV2Filters = {
  tf: 'M15' | 'H1' | 'ALL'
  exit: ResearchExit
  ml_only: boolean
  first_touch: boolean
  min_disp_h1: number
  sessions: string[]
  killzones: string[]
  fvg_only: boolean
  confluence_only: boolean
  direction: 'both' | 'bull' | 'bear'
  va_filter: boolean
  act_filter: boolean
}

export type ResearchV2Report = {
  symbol: string
  tf: string
  settings: {
    swing_n: number | null
    min_disp: number | null
    exit: ResearchExit
    exit_label: string
    commission: number
    slippage_pips: number
    net: boolean
    period: string
    ml_period_only: boolean
    va_filter?: boolean
    va_max_pos?: number
    act_filter?: boolean
    act_min?: number
  }
  break_even_win_pct: number | null
  all: Stats
  filtered: Stats
  equity_all: { time: number; value: number }[]
  equity_filtered: { time: number; value: number }[]
  by_session: BreakdownRow[]
  by_killzone: BreakdownRow[]
  by_disp: BreakdownRow[]
  by_year: BreakdownRow[]
  by_direction: BreakdownRow[]
}

export type BotConfig = {
  swing_n: number
  core_symbols: string[]
  disp_by_symbol: Record<string, number>
  default_disp: number
}

type Params = Record<string, string | number | undefined>

function buildUrl(path: string, params: Params): URL {
  const url = new URL(API + path)
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== '') url.searchParams.set(k, String(v))
  }
  return url
}

async function request<T>(path: string, params: Params = {}, method = 'GET'): Promise<T> {
  const res = await fetch(buildUrl(path, params), { method })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}


export type LiveTrade = {
  symbol: string
  direction: 'bull' | 'bear'
  fill_ts: number | null
  exit_ts: number | null
  entry: number
  exit: 'tp1' | 'trail'
  exit_reason: string
  r: number
  lots: number
  commission_usd?: number
  pnl: number
}

export type LiveOpenTrade = {
  symbol: string
  direction: 'bull' | 'bear'
  entry: number
  sl: number
  sl_pips: number
  lots: number
  risk_usd: number
  risk_pct: number | null
  exit: 'tp1' | 'trail'
  score: number
  r_now: number | null
  commission_usd?: number
  fill_time: number | null
}

export type LiveDecision = {
  bar_ts: number | null
  symbol: string
  direction: 'bull' | 'bear'
  entry: number
  score: number
  decision: string
  reason: string | null
  lots: number | null
  exit: string | null
}

export type LiveWhatIf = { decision: string; zones: number; win_pct: number; net_brain: number; net_tp1: number; net_trail: number }

export type LiveHealth = {
  updated: string
  started_at: string
  age_sec: number
  uptime_sec: number
  pid: number
  mt5?: {
    connected: boolean
    trade_allowed: boolean
    ping_ms: number | null
    build: number | null
    login: number | null
    server: string | null
    balance: number | null
    equity: number | null
    currency: string | null
    broker_time: string
    broker_time_checked: boolean
  }
  prices: Record<string, { bid: number; ask: number; spread_pips: number; tick_age_sec: number | null }>
  market_open: boolean
  system: {
    cpu_pct?: number
    ram_used_mb?: number
    ram_total_mb?: number
    disk_free_gb?: number
    disk_total_gb?: number
    bot_ram_mb?: number
    mt5_ram_mb?: number | null
  }
  restarts: { starts: number | null; last: string | null }
  news_blocked?: Record<string, string | null>
  news?: {
    block_before_min: number | null
    block_after_min: number | null
    events: { time: number; currency: string; event: string; pairs: string[] }[]
  }
}

export type LiveWatchZone = {
  direction: 'bull' | 'bear'
  entry: number
  sl: number
  sl_pips: number
  score: number
  exit: 'tp1' | 'trail'
  exit_bot?: 'tp1' | 'trail' | null   // what the exit brain picked (exit = yours if you changed it)
  user_exit?: string                  // '' = brain's choice, else the exit you set
  taken?: boolean                     // you pressed TAKE (ML low / value area / quiet overridden)
  lots: number
  status: string
  reason: string
  zone_candle_time: number | null
  created_at: number | null
  top: number | null
  bottom: number | null
  body_top: number | null
  body_bottom: number | null
  disp_atr: number | null
  has_fvg: boolean
}

export type LiveWatch = {
  updated: string
  threshold: number
  symbols: Record<string, { bar_time: number; zones: LiveWatchZone[] }>
}

// Halka live data (har ~1 s): bid/ask, chal rahi M5 candle, khule trades -> live P&L
export type LiveTicks = {
  status?: string
  error?: string
  updated: string | null
  age_sec: number | null
  prices: Record<string, { bid: number; ask: number; spread_pips: number; tick_age_sec?: number | null }>
  forming: Record<string, [number, number, number, number, number]>
  open: LiveTickTrade[]
  avoid?: { symbol: string; zone: number; direction: 'bull' | 'bear' }[]
  take?: { symbol: string; zone: number; direction: 'bull' | 'bear' }[]
  exit_set?: { symbol: string; zone: number; direction: 'bull' | 'bear'; exit: 'tp1' | 'trail' }[]
  close_pending?: { symbol: string; fill_time: number }[]
  exit_pending?: { symbol: string; fill_time: number; exit: 'tp1' | 'trail' }[]
  orb_close_pending?: { symbol: string; fill_time: number }[]
  exec?: ExecStatus | null
}

// REAL DEMO orders (src/live/execution.py -> data/live/exec_status.json)
export type RealLink = { status: string; order?: number | null; position?: number | null; real_entry?: number | null; entry_slip_pips?: number | null; error?: string | null }
export type ExecStatus = {
  updated: string
  age_sec?: number
  switch: { zone: boolean; orb: boolean }
  demo: boolean
  account: { login?: number; server?: string; demo?: boolean; balance?: number; equity?: number; margin_free?: number; currency?: string }
  positions: { strategy: string; ticket: number; symbol: string; buy: boolean; lots: number; price_open: number; sl: number; tp: number; profit: number }[]
  orders: { strategy: string; ticket: number; symbol: string; price: number; sl: number; tp: number; lots: number }[]
  links: { zone?: Record<string, RealLink>; orb?: Record<string, RealLink> }
  errors: string[]
}
export type RealSummary = {
  strategies: Record<string, { trades: number; real_r: number; shadow_r: number; r_gap: number | null; avg_entry_slip_pips: number | null; net_usd: number; commission_usd: number; wins: number }>
  trades: { strategy: string; symbol: string; direction: string; shadow_entry: number | null; real_entry: number | null; entry_slip_pips: number | null; shadow_r: number | null; real_r: number | null; real_exit_reason: string | null; net_usd: number | null; real_exit_time: string | null; note: string | null }[]
}

export type WeeklyBox = { n: number; R: number; pf: number | null; win_pct: number | null; note?: string }
type WeeklyScope = {
  bot_taken?: WeeklyBox
  you_avoided?: WeeklyBox
  you_taken?: WeeklyBox
  ml_low_skipped?: WeeklyBox
  your_exit_changes?: { n: number; R_yours: number; R_brain: number; R_gained: number; right: number; wrong: number }
  your_trade_exit_changes?: { n: number; matched: number; R_yours?: number; R_brain?: number; R_gained?: number; right?: number; wrong?: number } | null
  your_closes?: { n: number; matched: number; R_gained: number }
}
export type WeeklyReport = {
  status?: string
  error?: string
  from_?: string
  to?: string
  zone_account?: WeeklyBox & { pnl_usd: number }
  orb_account?: WeeklyBox & { pnl_usd: number; manual_closes: number }
  zone_week?: WeeklyScope
  zone_all_time?: WeeklyScope
  your_actions_week?: Record<string, number>
  real_week?: Record<string, { n: number; real_R: number; shadow_R: number; net_usd: number; avg_slip_pips: number | null }>
  text?: string
  history?: string[]
}

export type LiveControl =
  | { action: 'avoid' | 'unavoid' | 'take' | 'untake' | 'unexit'; symbol: string; zone: number; direction: 'bull' | 'bear' }
  | { action: 'exit'; symbol: string; zone: number; direction: 'bull' | 'bear'; exit: 'tp1' | 'trail' }
  | { action: 'close' | 'orb_close'; symbol: string; fill_time: number }
  | { action: 'trade_exit'; symbol: string; fill_time: number; exit: 'tp1' | 'trail' }
  | { action: 'exec_on' | 'exec_off'; symbol: 'ZONE' | 'ORB' }

export type LiveTickTrade = {
  symbol: string
  direction: 'bull' | 'bear'
  entry: number
  sl: number
  lots: number
  commission_usd?: number
  fill_time: number | null
  filled_at?: number | null
  exit?: string
  sl_pips?: number | null
  score?: number | null
  tp?: number | null
  stop_now?: number | null
  best_r?: number | null
  r_now?: number | null
  kind?: 'zone' | 'orb'          // 'orb' = ORB shadow trade (separate $10k account, TP 2R)
  exit_bot?: string | null       // exit brain's choice (exit = yours if you changed it)
  user_exit?: string             // '' = brain's choice
  taken_manual?: boolean         // you pressed TAKE on this zone
  zone?: number | null
  real?: RealLink                // real DEMO order of this trade (if execution is ON)
  range?: string | null          // ORB range (ASIA / LON30 / LON60)
  risk_usd?: number | null
}

// ---------------- ORB shadow (src/live/orb.py) ----------------
export type OrbRange = {
  day: string
  symbol: string
  range: 'ASIA' | 'LON30' | 'LON60'
  status: string
  status_text: string
  reason: string
  t0: number
  t1: number
  w_end: number
  eod: number
  hi: number | null
  lo: number | null
  w_ratio: number | null
  nr4: number | null
  tight: number | null
  direction: 'bull' | 'bear' | null
  break_time: number | null
  entry: number | null
  sl: number | null
  score: number | null
  open: boolean
  tp: number | null
  r_now: number | null
  exit_time?: number | null      // research: when the trade closed
  r?: number | null              // research: gross R at TP 2R (also what-if for skipped breaks)
  r_net?: number | null
  exit_reason?: string | null
}

export type OrbGroupStat = { n: number; pf: number | null; win: number; r: number; pnl?: number; range?: string; symbol?: string; year?: number }

export type OrbLive = {
  status: string
  source?: string
  error?: string
  updated?: number | null
  age_sec?: number | null
  threshold?: number
  tight_q30?: Record<string, number>
  day?: string | null
  account?: { capital: number; equity: number; peak: number; pnl: number; total_pct: number; dd_pct: number; risk_pct: number; n_closed: number }
  open?: { day: string; symbol: string; range: string; direction: 'bull' | 'bear'; entry: number; sl: number; tp: number; lots: number; risk_usd: number; score: number; r_now: number | null; fill_time: string }[]
  ranges?: OrbRange[]
  trades?: { day: string; symbol: string; range: string; direction: 'bull' | 'bear'; fill_time: string; exit_time: string; entry: number; sl: number; tp: number; score: number; lots: number; r: number; exit_reason: string; pnl: number; equity: number }[]
  live_stats?: { total?: OrbGroupStat; by_range_pair?: OrbGroupStat[]; by_range?: OrbGroupStat[]; by_symbol?: OrbGroupStat[] }
  equity_curve?: [number, number][]
}

export type OrbStat = { n: number; pf: number | null; win: number | null; r: number; avg_r: number | null }
export type OrbResearchFilters = {
  pairs: string[]; ranges: string[]; comp: string; ml: boolean; tp: string; one_per_day: boolean; direction: string; y0: number; y1: number
}
export type OrbResearchReport = {
  filters: OrbResearchFilters & { years: number[]; ml_years_note: string | null }
  total: OrbStat & { per_year: number; max_dd_r: number; pf_first_half: number | null; pf_second_half: number | null }
  baseline: OrbStat & { label: string }
  by_year: (OrbStat & { year: number })[]
  by_range_pair: (OrbStat & { range: string; symbol: string })[]
  by_symbol: (OrbStat & { symbol: string })[]
  by_range: (OrbStat & { range: string })[]
  by_weekday: (OrbStat & { weekday: string })[]
  equity: [number, number][]
}
export type OrbBrain = {
  meta: { trained_at: string; rows: number; data_to: string; threshold: number; keep: number; tp_r: number; tight_q30: Record<string, number>; label: string }
  importance: { feature: string; share: number }[]
  buckets: (OrbStat & { bucket: number; score_from: number; score_to: number })[]
  split: Record<'all_breaks' | 'ml_keep' | 'ml_drop' | 'ml_keep_comp' | 'ml_keep_nocomp' | 'comp_no_ml', OrbStat>
  note: string
}

export type OrbBacktest = {
  source: string
  total: OrbGroupStat
  max_dd_r: number
  by_range_pair: OrbGroupStat[]
  by_range: OrbGroupStat[]
  by_symbol: OrbGroupStat[]
  by_year: OrbGroupStat[]
  equity: [number, number][]
  account?: {
    risk_pct: number; capital: number; final: number; total_pct: number; cagr_pct: number; max_dd_pct: number
    worst_month_pct: number; worst_day_pct: number; losing_months_pct: number; longest_losing_streak: number
    yearly_return_pct: Record<string, number>; equity: [number, number][]
  }
  model?: { trained_at: string; rows: number; data_to: string; threshold: number; keep: number; tp_r: number; tight_q30: Record<string, number> }
}

export type LiveChart = {
  symbol: string
  tf: string
  status?: string
  error?: string
  bar_time?: number
  forming_time?: number
  candles: Omit<Candle, 'spread'>[]
  zones: LiveWatchZone[]
  open: {
    direction: 'bull' | 'bear'
    entry: number
    sl: number
    tp: number | null
    stop_now: number | null
    best_r: number | null
    exit: string
    lots: number
    r_now: number | null
    fill_time: number | null
    kind?: 'zone' | 'orb'
    range?: string | null
    zone?: number | null
    real_entry?: number | null     // real DEMO fill price (src/live/execution.py)
  }[]
  trades: {
    direction: 'bull' | 'bear'; entry: number; sl: number; exit: string; r: number; exit_reason: string; pnl: number; fill_ts: number; exit_ts: number
    kind?: 'zone' | 'orb'; range?: string | null; tp?: number | null; exit_px?: number | null; lots?: number | null
  }[]
  orb?: OrbRange[]
  traded_zones?: {
    direction: 'bull' | 'bear'
    top: number | null
    bottom: number | null
    body_top: number | null
    body_bottom: number | null
    entry?: number | null
    created_at: number | null
    zone_candle_time: number | null
    disp_atr: number | null
    has_fvg: boolean
    fill_time: number
    exit_ts: number | null
  }[]
}

export type LiveStatus = {
  status: 'running' | 'stopped' | 'none' | 'unreachable' | 'market_closed'
  real?: RealSummary
  health?: LiveHealth | null
  log?: string[]
  watch?: LiveWatch | null
  source?: 'local' | 'vps'
  error?: string
  updated?: number | null
  age_sec?: number | null
  full_state?: boolean
  account?: {
    capital: number
    equity: number
    peak: number
    dd_pct: number
    total_pct: number
    total_usd?: number
    drawdown_usd?: number
    commission_per_lot?: number
    commission_total?: number
    commission_closed?: number
    commission_open?: number
    today_pnl: number
    month_pnl: number
    kill: boolean
    braked: boolean
    risk_now_pct: number
    brake_dd_pct: number
    kill_dd_pct: number
    daily_loss_pct: number
    daily_budget_pct: number
    combo_enabled?: boolean
    risk_mode?: 'aggressive' | 'conservative'
    base_risk_pct?: number
    risk_cap_pct?: number
  }
  open_trades?: LiveOpenTrade[]
  closed?: { count: number; wins: number; win_pct: number | null; pnl: number; r_sum: number; pf: number | null; commission?: number }
  curve?: { time: number; value: number }[]
  trades?: LiveTrade[]
  decisions?: LiveDecision[]
  whatif?: LiveWhatIf[]
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),
  symbols: () => request<Record<string, string[]>>('/api/symbols'),
  candles: (p: { symbol: string; tf: string; start?: string; end?: string; limit?: number }) =>
    request<Candle[]>('/api/candles', p),
  zones: (p: { symbol: string; tf: string; start?: string; end?: string; swing_n: number; min_disp: number }) =>
    request<Zone[]>('/api/zones', p),
  scan: (p: { symbol: string; tf: string; swing_n: number; min_disp: number }) =>
    request<{ started: boolean }>('/api/scan', p, 'POST'),
  researchBuild: (p: { symbol: string; tf: string; swing_n: number; min_disp: number }) =>
    request<{ started: boolean }>('/api/research/build', p, 'POST'),
  researchReport: (symbol: string, tf: string, f: ResearchFilters) =>
    request<ResearchReport>('/api/research/report', {
      symbol, tf,
      min_disp_h1: f.min_disp_h1,
      sessions: f.sessions.join(','),
      fvg_only: String(f.fvg_only),
      confluence_only: String(f.confluence_only),
      direction: f.direction,
    }),
  mlTrain: (symbols: string[]) => request<{ started: boolean }>('/api/ml/train', { symbols: symbols.join(',') }, 'POST'),
  mlReport: () => request<MLReport>('/api/ml/report'),
  mlScores: (p: { symbol: string; tf: string; start?: string; end?: string }) => request<MLScore[]>('/api/ml/scores', p),
  riskReport: () => request<RiskReport>('/api/risk/report'),
  config: () => request<BotConfig>('/api/config'),
  live: () => request<LiveStatus>('/api/live'),
  liveChart: (symbol: string, tf: string) => request<LiveChart>('/api/live/chart', { symbol, tf }),
  liveTicks: () => request<LiveTicks>('/api/live/ticks'),
  liveOrb: () => request<OrbLive>('/api/live/orb'),
  orbBacktest: () => request<OrbBacktest>('/api/orb/backtest'),
  orbResearchReport: (f: OrbResearchFilters) =>
    request<OrbResearchReport>('/api/orb/research/report', {
      pairs: f.pairs.join(','), ranges: f.ranges.join(','), comp: f.comp, ml: String(f.ml), tp: f.tp,
      one_per_day: String(f.one_per_day), direction: f.direction, y0: f.y0, y1: f.y1,
    }),
  orbResearchChart: (symbol: string, start: string, end: string) =>
    request<{ symbol: string; ranges: OrbRange[] }>('/api/orb/research/chart', { symbol, start, end }),
  orbResearchBrain: () => request<OrbBrain>('/api/orb/research/brain'),
  liveControl: (c: LiveControl) => request<{ ok: boolean }>('/api/live/control', { ...c }, 'POST'),
  liveWeekly: () => request<WeeklyReport>('/api/live/weekly'),
  researchV2Build: (symbol: string) => request<{ started: boolean }>('/api/research/v2/build', { symbol }, 'POST'),
  researchV2Report: (symbol: string, f: ResearchV2Filters) =>
    request<ResearchV2Report>('/api/research/v2/report', {
      symbol,
      tf: f.tf,
      exit: f.exit,
      ml_only: String(f.ml_only),
      first_touch: String(f.first_touch),
      min_disp_h1: f.min_disp_h1,
      sessions: f.sessions.join(','),
      killzones: f.killzones.join(','),
      fvg_only: String(f.fvg_only),
      confluence_only: String(f.confluence_only),
      direction: f.direction,
      va_filter: String(f.va_filter),
      act_filter: String(f.act_filter),
    }),
}
