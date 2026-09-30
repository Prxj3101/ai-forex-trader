import type { RobotVariant } from '../components/RobotHead'
import type { AgentEvent, AgentId } from './api'
import type { Theme } from './theme'

export type AgentInfo = { id: AgentId; name: string; color: string; tip: string; pipeline: string; robot: RobotVariant }

// Humare system ke layers = agents. Order = decision pipeline ka order.
export const AGENTS: AgentInfo[] = [
  { id: 'data', name: 'DATA AGENT', color: '#00f0ff', tip: 'Downloads, cleans and resamples market data', pipeline: 'new candle', robot: 'dish' },
  { id: 'structure', name: 'STRUCTURE AGENT', color: '#b98cff', tip: 'Detects swings, BOS and CHoCH', pipeline: 'BOS / CHoCH', robot: 'twin' },
  { id: 'zone', name: 'ZONE HUNTER', color: '#00ff88', tip: 'Finds orderflow zones, FVGs and liquidity', pipeline: 'orderflow zone', robot: 'antenna' },
  { id: 'ml', name: 'ML BRAIN', color: '#ffc400', tip: 'Scores each zone: probability it works', pipeline: 'probability', robot: 'dome' },
  { id: 'risk', name: 'RISK AGENT', color: '#ff5cc8', tip: 'Position size, stop loss, daily loss limit', pipeline: 'lot + SL', robot: 'crest' },
  { id: 'execution', name: 'EXECUTION AGENT', color: '#ff2e63', tip: 'Sends orders to the demo account', pipeline: 'order', robot: 'bolt' },
]

/** Theme ke hisaab se agents ke colors: Data agent = accent, baaki overrides ke saath. */
export function agentsFor(theme: Theme): AgentInfo[] {
  return AGENTS.map((a) => ({
    ...a,
    color: a.id === 'data' ? theme.accent : theme.overrides?.[a.id] ?? a.color,
  }))
}

export const agentColor = (theme: Theme, id: AgentId) =>
  agentsFor(theme).find((a) => a.id === id)?.color ?? theme.accent
export const agentName = (id: AgentId) => AGENTS.find((a) => a.id === id)?.name ?? 'SYSTEM'

export type AgentState = { status: 'active' | 'done' | 'idle' | 'error' | 'offline'; message: string; progress: number }

// ML, Risk, Execution abhi bane nahi hain -> OFFLINE dikhayenge (jhooth nahi bolenge)
const NOT_BUILT: AgentId[] = ['ml', 'risk', 'execution']

/** Har agent ka latest event dekh ke uska abhi ka state nikalta hai. */
export function agentStates(events: AgentEvent[], now: number): Record<string, AgentState> {
  const out: Record<string, AgentState> = {}
  for (const a of AGENTS) {
    if (NOT_BUILT.includes(a.id)) {
      out[a.id] = { status: 'offline', message: 'Not built yet', progress: 0 }
      continue
    }
    const last = [...events].reverse().find((e) => e.agent === a.id)
    if (!last) {
      out[a.id] = { status: 'idle', message: 'Waiting for a task', progress: 0 }
      continue
    }
    const age = now / 1000 - last.ts
    let status: AgentState['status'] = last.status === 'error' ? 'error' : last.status === 'active' ? 'active' : 'done'
    if (status === 'active' && age > 30) status = 'idle'
    if (status === 'done' && age > 6) status = 'idle'
    out[a.id] = { status, message: last.message, progress: last.progress ?? (status === 'done' ? 100 : 0) }
  }
  return out
}
