import { useEffect, useState } from 'react'
import { api, type OrbLive, type OrbRange } from '../lib/api'
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
  const open = d.open ?? []
  const trades = d.trades ?? []
  const stale = d.age_sec != null && d.age_sec > 300

  return (
    <div className="panel">
      {header}
      <div className="mb-3 grid grid-cols-3 gap-x-3 gap-y-2.5 border p-2.5 sm:grid-cols-6" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
        <Stat label="ORB EQUITY" value={usd(a.equity)} color="var(--cyan)" tip={`Separate paper account, started with ${usd(a.capital)}. Closed trades only.`} />
        <Stat label="TOTAL" value={`${usd(a.pnl)} (${a.total_pct > 0 ? '+' : ''}${a.total_pct.toFixed(2)}%)`} color={pnlColor(a.pnl)} tip="After $5/lot commission and stop slippage" />
        <Stat label="DRAWDOWN" value={`-${a.dd_pct.toFixed(2)}%`} color={a.dd_pct > 5 ? 'var(--amber)' : 'var(--text)'} tip={`From the peak ${usd(a.peak)}`} />
        <Stat label="TRADES" value={`${a.n_closed}`} tip="Closed ORB trades" />
        <Stat label="WIN / PF" value={ls ? `${ls.win}% / ${ls.pf ?? '—'}` : '—'} color={ls?.pf != null ? (ls.pf >= 1 ? 'var(--green)' : 'var(--red)') : undefined} tip="Win rate and profit factor of closed ORB trades (backtest: ~44% / 1.16)" />
        <Stat label="RISK / TRADE" value={`${a.risk_pct}%`} tip={`ML needs ${d.threshold != null ? Math.round(d.threshold * 100) : '—'}% (top 30%). TP 2R, SL other side of the range, close 16:00 NY, max 1 ORB per pair per day.`} />
      </div>
      {stale && <div className="mb-2 text-[10px]" style={{ color: 'var(--amber)' }}>ORB state not updated for {Math.round((d.age_sec ?? 0) / 60)} min: is the bot running?</div>}

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
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
        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">OPEN ORB TRADES ({open.length})</div>
          {open.length === 0 ? <div className="mb-3 text-[10px] text-muted">No open ORB trades.</div> : (
            <table className="mb-3 w-full text-[10px]">
              <thead className="text-[9px] text-muted"><tr>
                <th className="text-left font-normal">PAIR</th><th className="text-left font-normal">RANGE</th><th className="text-left font-normal">SIDE</th>
                <th className="text-right font-normal">ENTRY</th><th className="text-right font-normal">SL</th><th className="text-right font-normal">TP</th>
                <th className="text-right font-normal">R NOW</th>
              </tr></thead>
              <tbody>{open.map((t) => {
                const dg = digitsOf(t.symbol)
                return (
                  <tr key={`${t.symbol}${t.fill_time}`} className="border-t" style={{ borderColor: 'var(--line)' }}>
                    <td>{t.symbol}</td><td style={{ color: ORB_COLORS[t.range] }}>{t.range}</td>
                    <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'} {t.lots}</td>
                    <td className="text-right">{t.entry.toFixed(dg)}</td><td className="text-right">{t.sl.toFixed(dg)}</td><td className="text-right">{t.tp.toFixed(dg)}</td>
                    <td className="text-right" style={{ color: pnlColor(t.r_now ?? 0) }}>{t.r_now != null ? `${t.r_now > 0 ? '+' : ''}${t.r_now.toFixed(2)}R` : '—'}</td>
                  </tr>
                )
              })}</tbody>
            </table>
          )}
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">LAST ORB TRADES</div>
          {trades.length === 0 ? <div className="text-[10px] text-muted">No closed ORB trades yet.</div> : (
            <table className="w-full text-[10px]">
              <thead className="text-[9px] text-muted"><tr>
                <th className="text-left font-normal">CLOSED (NY)</th><th className="text-left font-normal">PAIR</th><th className="text-left font-normal">RANGE</th>
                <th className="text-left font-normal">SIDE</th><th className="text-left font-normal">EXIT</th><th className="text-right font-normal">R</th><th className="text-right font-normal">P&amp;L</th>
              </tr></thead>
              <tbody>{trades.slice(0, 8).map((t) => (
                <tr key={`${t.symbol}${t.fill_time}`} className="border-t" style={{ borderColor: 'var(--line)' }}>
                  <td className="text-muted">{fmtTs(Date.parse(t.exit_time) / 1000)}</td><td>{t.symbol}</td>
                  <td style={{ color: ORB_COLORS[t.range] }}>{t.range}</td>
                  <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'}</td>
                  <td className="text-muted">{t.exit_reason.toUpperCase()}</td>
                  <td className="text-right" style={{ color: pnlColor(t.r) }}>{t.r > 0 ? '+' : ''}{t.r.toFixed(2)}</td>
                  <td className="text-right" style={{ color: pnlColor(t.pnl) }}>{usd(t.pnl)}</td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
