import { useEffect, useState } from 'react'
import { api, type LiveTickTrade, type OrbLive, type OrbRange } from '../lib/api'
import { floatPnl, fmtFill, fmtSigned, fmtUsd, useLiveTicks } from '../lib/pnl'
import { ORB_COLORS } from '../lib/orbPrimitive'
import { fmtTs } from '../lib/time'

/*
  ORB SHADOW panel (LIVE tab): aaj ke Asia / London dabbe har pair pe, unka status (kyun trade hua / nahi hua),
  khule ORB trades aur alag $10k paper account. Data: /api/live/orb (VPS bot ka data/live/orb_state.json).
*/

const POLL_MS = 5_000
const usd = (v: number) => `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
const pnlColor = (v: number) => (v > 0 ? 'var(--green)' : v < 0 ? 'var(--red)' : 'var(--text)')
const digitsOf = (s: string) => (s.endsWith('JPY') ? 3 : 5)
const COMM_PER_LOT = 5                                  // src/live/orb.py: $5 / lot round trip
const pipOf = (s: string) => (s.endsWith('JPY') ? 0.01 : 0.0001)
const hm = (sec: number) => new Date(sec * 1000).toLocaleTimeString('en-GB', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false })

export const ORB_STATUS_COLOR: Record<string, string> = {
  waiting: 'var(--muted)', forming: 'var(--cyan)', armed: 'var(--green)', traded: 'var(--cyan)',
  skip_nocomp: 'var(--muted)', no_break: 'var(--muted)', skip_ml: 'var(--amber)', skip_daily: 'var(--amber)',
  skip_lot: 'var(--amber)', no_data: 'var(--red)',
}
const STATUS_TIP: Record<string, string> = {
  waiting: 'The range has not started yet',
  forming: 'The range is being built right now (high / low still moving)',
  armed: 'Market was compressed (NR4 or tight range): buy stop above the high and sell stop below the low are ready until 11:00 NY',
  traded: 'The range broke, ML liked it: ORB trade opened (SL other side, TP 2R, close at 16:00 NY)',
  skip_nocomp: 'No compression (yesterday not NR4 and range not tight): no ORB trade on this range today',
  no_break: 'The range was not broken before 11:00 NY',
  skip_ml: 'The range broke, but the ML score was below the top-30% threshold',
  skip_daily: 'ML liked it, but this pair already had an ORB trade today (max 1 per pair per day)',
  skip_lot: 'Lot size below 0.01',
  no_data: 'Not enough candles for this range',
}

function Stat({ label, value, color, tip }: { label: string; value: string; color?: string; tip: string }) {
  return (
    <div title={tip} className="cursor-help">
      <small className="block text-[9px] tracking-wider text-muted">{label}</small>
      <b className="font-display text-[15px] font-medium" style={{ color: color ?? 'var(--text)' }}>{value}</b>
    </div>
  )
}

function RangeRow({ r }: { r: OrbRange }) {
  const dg = digitsOf(r.symbol)
  const c = ORB_STATUS_COLOR[r.status] ?? 'var(--muted)'
  return (
    <tr className="border-t" style={{ borderColor: 'var(--line)' }}>
      <td className="py-0.5">{r.symbol}</td>
      <td style={{ color: ORB_COLORS[r.range] }}>{r.range}</td>
      <td className="text-muted">{hm(r.t0)}-{hm(r.t1)}</td>
      <td className="text-right">{r.hi != null ? r.hi.toFixed(dg) : '—'}</td>
      <td className="text-right">{r.lo != null ? r.lo.toFixed(dg) : '—'}</td>
      <td className="text-right" title="Range size vs the average of the last 20 same ranges (tight = small)">{r.w_ratio != null ? `${r.w_ratio.toFixed(2)}x` : '—'}</td>
      <td className="text-center" title="NR4: yesterday's daily range was the smallest of the last 4 days">{r.nr4 == null ? '—' : r.nr4 ? '✓' : '·'}</td>
      <td className="text-right" title="ML score at the break (needs the top-30% threshold)">{r.score != null ? `${Math.round(r.score * 100)}%` : '—'}</td>
      <td className="text-right"><span className="chip !px-1.5 !py-0 text-[9px]" style={{ color: c, borderColor: c }} title={STATUS_TIP[r.status] ?? r.status}>{r.status_text}</span></td>
      <td className="max-w-[340px] truncate pl-2 text-[9px] text-muted" title={r.reason}>{r.reason}</td>
    </tr>
  )
}

export default function OrbPanel({ connected }: { connected: boolean }) {
  const [d, setD] = useState<OrbLive | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!connected) return
    let alive = true
    const load = () =>
      api.liveOrb()
        .then((x) => { if (alive) { setD(x); setErr(null) } })
        .catch((e: unknown) => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
    void load()
    const id = window.setInterval(load, POLL_MS)
    return () => { alive = false; window.clearInterval(id) }
  }, [connected])

  const ticks = useLiveTicks(connected && (d?.open?.length ?? 0) > 0)   // live P&L sirf jab ORB trade khula ho

  const header = (
    <div className="ptitle flex-wrap">
      <span>ORB SHADOW</span>
      <em title="Opening range breakout: Asia (19:00-02:00 NY) and London (03:00-03:30 / 04:00) ranges. Trades only when the market was compressed (NR4 or tight range) and the ML score is in the top 30%. Separate $10,000 paper account, 0.4% risk per trade, TP 2R or close at 16:00 NY.">
        ASIA / LONDON RANGE BREAKOUT · COMPRESSION + ML · SEPARATE PAPER ACCOUNT
      </em>
    </div>
  )

  if (!d || d.status !== 'ok' || !d.account) {
    return (
      <div className="panel">
        {header}
        <div className="py-4 text-center text-[11px] text-muted">
          {err ? `Could not load ORB data: ${err}`
            : d?.status === 'unreachable' ? (d.error ?? 'Cannot reach the VPS.')
              : 'No ORB data yet. It starts with the new bot version (git pull + restart FOREX BOT on the VPS).'}
        </div>
      </div>
    )
  }

  const a = d.account
  const ls = d.live_stats?.total
  const ranges = d.ranges ?? []
  const trades = d.trades ?? []
  const stale = d.age_sec != null && d.age_sec > 300
  // khule ORB trades: ticks (har ~1 s, live P&L) se; ticks na hon toh ORB state (5 s)
  const tickOpen = (ticks?.updated ? ticks.open : []).filter((t) => t.kind === 'orb')
  const open: LiveTickTrade[] = !(d.open?.length) ? []                 // ORB state: koi trade khula nahi
    : tickOpen.length ? tickOpen
    : (d.open ?? []).map((t) => ({ ...t, kind: 'orb' as const, exit: 'tp2', fill_time: Date.parse(t.fill_time) / 1000, commission_usd: t.lots * COMM_PER_LOT }))
  const live = open.map((t) => {
    const p = ticks?.prices[t.symbol]
    return p ? floatPnl(t, p.bid, p.ask) : null
  })
  const floating = live.reduce((s, p) => s + (p?.usd ?? 0), 0)
  const hasLive = live.some((p) => p !== null)
  const closedComm = trades.reduce((s, t) => s + t.lots * COMM_PER_LOT, 0)
  const openComm = open.reduce((s, t) => s + (t.commission_usd ?? t.lots * COMM_PER_LOT), 0)

  return (
    <div className="panel">
      {header}
      <div className="mb-3 grid grid-cols-3 gap-x-3 gap-y-2.5 border p-2.5 sm:grid-cols-4 xl:grid-cols-8" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
        <Stat label="ORB EQUITY" value={usd(a.equity)} color="var(--cyan)" tip={`Separate paper account, started with ${usd(a.capital)}. Closed trades only.`} />
        <Stat label="TOTAL" value={`${usd(a.pnl)} (${a.total_pct > 0 ? '+' : ''}${a.total_pct.toFixed(2)}%)`} color={pnlColor(a.pnl)} tip="After $5/lot commission and stop slippage" />
        <Stat label="FLOATING" value={open.length ? (hasLive ? fmtUsd(floating) : '—') : '$0.00'} color={pnlColor(floating)} tip="Live profit / loss of the open ORB trades right now (after commission). Updates about every second." />
        <Stat label="DRAWDOWN" value={`-${a.dd_pct.toFixed(2)}%`} color={a.dd_pct > 5 ? 'var(--amber)' : 'var(--text)'} tip={`From the peak ${usd(a.peak)}`} />
        <Stat label="COMM. TOTAL" value={usd(-(closedComm + openComm))} color="var(--red)" tip={`Round-trip commission: ${usd(-closedComm)} on the last ${trades.length} closed trades + ${usd(-openComm)} for open trades`} />
        <Stat label="TRADES" value={`${a.n_closed}`} tip="Closed ORB trades" />
        <Stat label="WIN / PF / R" value={ls ? `${ls.win}% / ${ls.pf ?? '—'} / ${ls.r > 0 ? '+' : ''}${ls.r}R` : '—'} color={ls?.pf != null ? (ls.pf >= 1 ? 'var(--green)' : 'var(--red)') : undefined} tip="Win rate, profit factor and total R of closed ORB trades (backtest: ~44% / 1.16)" />
        <Stat label="RISK / TRADE" value={`${a.risk_pct}%`} tip={`ML needs ${d.threshold != null ? Math.round(d.threshold * 100) : '—'}% (top 30%). TP 2R, SL other side of the range, close 16:00 NY, max 1 ORB per pair per day.`} />
      </div>
      {stale && <div className="mb-2 text-[10px]" style={{ color: 'var(--amber)' }}>ORB state not updated for {Math.round((d.age_sec ?? 0) / 60)} min: is the bot running?</div>}

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div className="min-w-0 overflow-x-auto">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">RANGES {d.day ? `· ${d.day} (NY)` : ''}</div>
          {ranges.length === 0 ? <div className="text-[10px] text-muted">No ranges yet today.</div> : (
            <table className="w-full text-[10px]">
              <thead className="text-[9px] text-muted"><tr>
                <th className="text-left font-normal">PAIR</th><th className="text-left font-normal">RANGE</th><th className="text-left font-normal">NY</th>
                <th className="text-right font-normal">HIGH</th><th className="text-right font-normal">LOW</th>
                <th className="text-right font-normal">SIZE</th><th className="font-normal">NR4</th><th className="text-right font-normal">ML</th>
                <th className="text-right font-normal">STATUS</th><th className="pl-2 text-left font-normal">WHY</th>
              </tr></thead>
              <tbody>{ranges.map((r) => <RangeRow key={`${r.day}${r.symbol}${r.range}`} r={r} />)}</tbody>
            </table>
          )}
        </div>
        <div className="min-w-0 overflow-x-auto">
          <div className="mb-1 flex items-center gap-2 text-[9px] tracking-[2px] text-muted">
            OPEN ORB TRADES ({open.length})
            {open.length > 0 && hasLive && (
              <span className="text-[11px] tracking-[1px]" style={{ color: pnlColor(floating) }} title="Floating profit / loss of all open ORB trades (after commission), about every second">· FLOATING {fmtUsd(floating)}</span>
            )}
            {open.length > 0 && hasLive && <i className="inline-block h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: 'var(--green)' }} title="Live prices" />}
          </div>
          {open.length === 0 ? <div className="mb-3 text-[10px] text-muted">No open ORB trades.</div> : (
            <table className="mb-3 w-full text-[10px]">
              <thead className="text-[9px] text-muted"><tr>
                <th className="text-left font-normal">PAIR</th><th className="text-left font-normal">RANGE</th><th className="text-left font-normal">SIDE</th>
                <th className="text-right font-normal">ENTRY</th><th className="text-right font-normal">SL</th><th className="text-right font-normal">TP</th>
                <th className="text-right font-normal" title="Live price (bid for a buy, ask for a sell)">NOW</th>
                <th className="text-right font-normal">PIPS</th>
                <th className="text-right font-normal" title="Round-trip commission for this trade">COMM.</th>
                <th className="text-right font-normal" title="Live profit / loss in dollars, after commission">P&amp;L</th>
                <th className="text-right font-normal" title="Live profit / loss in R (1R = stop distance)">R</th>
              </tr></thead>
              <tbody>{open.map((t, i) => {
                const dg = digitsOf(t.symbol)
                const p = live[i]
                const px = ticks?.prices[t.symbol]
                const now = px ? (t.direction === 'bull' ? px.bid : px.ask) : null
                const r = p ? p.r : t.r_now
                const comm = t.commission_usd ?? t.lots * COMM_PER_LOT
                return (
                  <tr key={`${t.symbol}${t.fill_time}`} className="border-t" style={{ borderColor: 'var(--line)' }}
                    title={`Filled ${fmtFill(t)} NY · SL ${t.sl.toFixed(dg)} (${(Math.abs(t.entry - t.sl) / pipOf(t.symbol)).toFixed(1)} pips)${t.risk_usd != null ? ` · risk ${usd(t.risk_usd)}` : ''}${t.score != null ? ` · ML ${Math.round(t.score * 100)}%` : ''} · TP 2R, closes 16:00 NY`}>
                    <td>{t.symbol}</td><td style={{ color: ORB_COLORS[t.range ?? ''] }}>{t.range}</td>
                    <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'} {t.lots}</td>
                    <td className="text-right">{t.entry.toFixed(dg)}</td><td className="text-right" style={{ color: 'var(--red)' }}>{t.sl.toFixed(dg)}</td>
                    <td className="text-right" style={{ color: 'var(--green)' }}>{t.tp != null ? t.tp.toFixed(dg) : '—'}</td>
                    <td className="text-right">{now != null ? now.toFixed(dg) : '—'}</td>
                    <td className="text-right" style={{ color: pnlColor(p?.pips ?? 0) }}>{p ? fmtSigned(p.pips) : '—'}</td>
                    <td className="text-right" style={{ color: 'var(--red)' }}>{usd(-comm)}</td>
                    <td className="text-right" style={{ color: pnlColor(p?.usd ?? 0) }}>{p ? fmtUsd(p.usd) : '—'}</td>
                    <td className="text-right" style={{ color: pnlColor(r ?? 0) }}>{r == null ? '—' : `${r > 0 ? '+' : ''}${r.toFixed(2)}R`}</td>
                  </tr>
                )
              })}</tbody>
            </table>
          )}
        </div>
      </div>

      <div className="mt-2 mb-1 text-[9px] tracking-[2px] text-muted">CLOSED ORB TRADES (LATEST {trades.length})</div>
      {trades.length === 0 ? <div className="text-[10px] text-muted">No closed ORB trades yet.</div> : (
        <div className="max-h-[230px] overflow-auto">
          <table className="w-full text-[10px]">
            <thead className="sticky top-0 text-[9px] text-muted" style={{ background: 'var(--panel)' }}><tr>
              <th className="text-left font-normal">OPENED (NY)</th><th className="text-left font-normal">CLOSED (NY)</th><th className="text-left font-normal">PAIR</th>
              <th className="text-left font-normal">RANGE</th><th className="text-left font-normal">SIDE</th><th className="text-right font-normal">LOTS</th>
              <th className="text-right font-normal">ENTRY</th><th className="text-right font-normal">SL</th><th className="text-right font-normal">TP</th>
              <th className="text-right font-normal" title="Exit price (from the R result)">EXIT</th><th className="text-left pl-2 font-normal">WHY</th>
              <th className="text-right font-normal">R</th><th className="text-right font-normal">COMM.</th><th className="text-right font-normal">P&amp;L</th>
              <th className="text-right font-normal" title="ORB account equity after this trade">EQUITY</th>
            </tr></thead>
            <tbody>{trades.map((t) => {
              const dg = digitsOf(t.symbol)
              const exitPx = t.entry + (t.direction === 'bull' ? 1 : -1) * t.r * Math.abs(t.entry - t.sl)
              return (
                <tr key={`${t.symbol}${t.fill_time}`} className="border-t" style={{ borderColor: 'var(--line)' }}
                  title={`ML ${Math.round(t.score * 100)}% · SL ${(Math.abs(t.entry - t.sl) / pipOf(t.symbol)).toFixed(1)} pips`}>
                  <td className="text-muted">{fmtTs(Date.parse(t.fill_time) / 1000)}</td>
                  <td className="text-muted">{fmtTs(Date.parse(t.exit_time) / 1000)}</td>
                  <td>{t.symbol}</td><td style={{ color: ORB_COLORS[t.range] }}>{t.range}</td>
                  <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'}</td>
                  <td className="text-right">{t.lots}</td>
                  <td className="text-right">{t.entry.toFixed(dg)}</td><td className="text-right">{t.sl.toFixed(dg)}</td><td className="text-right">{t.tp.toFixed(dg)}</td>
                  <td className="text-right">{exitPx.toFixed(dg)}</td>
                  <td className="pl-2 text-muted">{t.exit_reason === 'tp' ? 'TP 2R' : t.exit_reason === 'sl' ? 'SL' : t.exit_reason === 'eod' ? '16:00 CLOSE' : t.exit_reason.toUpperCase()}</td>
                  <td className="text-right" style={{ color: pnlColor(t.r) }}>{t.r > 0 ? '+' : ''}{t.r.toFixed(2)}</td>
                  <td className="text-right" style={{ color: 'var(--red)' }}>{usd(-t.lots * COMM_PER_LOT)}</td>
                  <td className="text-right" style={{ color: pnlColor(t.pnl) }}>{usd(t.pnl)}</td>
                  <td className="text-right text-muted">{usd(t.equity)}</td>
                </tr>
              )
            })}</tbody>
          </table>
        </div>
      )}
    </div>
  )
}
