import { useEffect, useState } from 'react'
import { api, type LiveControl, type LiveTicks } from './api'

/** Dashboard button -> bot (AVOID / UNDO / CLOSE). Pehle confirm poochta hai. true = bheja gaya. */
export async function sendControl(c: LiveControl, question: string | null): Promise<boolean> {
  if (question && !window.confirm(question)) return false
  try {
    await api.liveControl(c)
    return true
  } catch (e) {
    window.alert(`Could not send the command: ${e instanceof Error ? e.message : String(e)}`)
    return false
  }
}

/*
  Khule trade ka CHALTA HUA (floating) profit / loss - MT5 jaisa, har ~1 sec.
  BUY band hota hai bid pe, SELL band hota hai ask pe. Commission ($ per lot, round trip) minus.
  Final P&L bot ke hisaab se thoda alag ho sakta hai (exit candle + slippage).
*/

export const COMMISSION_PER_LOT = 5        // src/config.py BROKER_COMMISSION (Exness Raw $5 round trip)
const LOT = 100_000

export type Pnl = { pips: number; usd: number; r: number }

export function pipSize(symbol: string) {
  return symbol.endsWith('JPY') ? 0.01 : 0.0001
}

export function floatPnl(
  t: { symbol: string; direction: 'bull' | 'bear'; entry: number; sl: number; lots: number },
  bid: number,
  ask: number,
): Pnl {
  const buy = t.direction === 'bull'
  const move = buy ? bid - t.entry : t.entry - ask              // price units, + = profit
  const exitPx = buy ? bid : ask
  // quote currency USD (EURUSD, GBPUSD) -> seedha $; USDJPY -> yen ko $ mein (exit bhav se)
  const usdMove = t.symbol.endsWith('USD') ? move * LOT * t.lots : (move * LOT * t.lots) / exitPx
  const usd = usdMove - COMMISSION_PER_LOT * t.lots
  const risk = Math.abs(t.entry - t.sl)
  return { pips: move / pipSize(t.symbol), usd, r: risk > 0 ? move / risk : 0 }
}

/** /api/live/ticks har `ms` poll (default 1 s). Bot band / VPS na mile toh null. */
export function useLiveTicks(connected: boolean, ms = 1000) {
  const [ticks, setTicks] = useState<LiveTicks | null>(null)
  useEffect(() => {
    if (!connected) return
    let alive = true
    let busy = false
    const load = () => {
      if (busy) return                       // pichli request abhi chal rahi hai -> skip (VPS door hai)
      busy = true
      api.liveTicks()
        .then((d) => { if (alive) setTicks(d.status === 'unreachable' ? null : d) })
        .catch(() => { if (alive) setTicks(null) })
        .finally(() => { busy = false })
    }
    load()
    const id = window.setInterval(load, ms)
    return () => { alive = false; window.clearInterval(id) }
  }, [connected, ms])
  return ticks
}

/** Trade kab khula (NY time, seconds ke saath). filled_at = asli ghadi ka time; purane trades mein sirf candle time. */
export function fmtFill(t: { filled_at?: number | null; fill_time: number | null }) {
  const s = t.filled_at ?? t.fill_time
  if (s == null) return '?'
  return new Date(s * 1000).toLocaleString('en-GB', {
    timeZone: 'America/New_York', day: '2-digit', month: 'short',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  })
}

/** Sirf HH:MM:SS (NY). */
export function fmtFillShort(t: { filled_at?: number | null; fill_time: number | null }) {
  const s = t.filled_at ?? t.fill_time
  if (s == null) return '?'
  return new Date(s * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/New_York', hour12: false })
}

export const fmtUsd = (v: number) => `${v < 0 ? '-' : '+'}$${Math.abs(v).toFixed(2)}`
export const fmtSigned = (v: number, d = 1) => `${v > 0 ? '+' : ''}${v.toFixed(d)}`
