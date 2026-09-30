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
  kind: 'log' | 'result' | 'research'
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
}
