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

export type LiveChart = {
  symbol: string
  tf: string
  status?: string
  error?: string
  bar_time?: number
  forming_time?: number
  candles: Omit<Candle, 'spread'>[]
  zones: LiveWatchZone[]
  open: { direction: 'bull' | 'bear'; entry: number; sl: number; exit: string; lots: number; r_now: number | null; fill_time: number | null }[]
  trades: { direction: 'bull' | 'bear'; entry: number; r: number; exit_reason: string; pnl: number; fill_ts: number; exit_ts: number }[]
}

export type LiveStatus = {
  status: 'running' | 'stopped' | 'none' | 'unreachable' | 'market_closed'
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
    today_pnl: number
    month_pnl: number
    kill: boolean
    braked: boolean
    risk_now_pct: number
    brake_dd_pct: number
    kill_dd_pct: number
    daily_loss_pct: number
    daily_budget_pct: number
  }
  open_trades?: LiveOpenTrade[]
  closed?: { count: number; wins: number; win_pct: number | null; pnl: number; r_sum: number; pf: number | null }
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
