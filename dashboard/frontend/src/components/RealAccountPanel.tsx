import { useEffect, useMemo, useState } from 'react'
import { api, type RealAccount, type RealClosed, type RealOpen, type RealStats } from '../lib/api'
import { floatPnl, fmtSigned, fmtUsd, useLiveTicks } from '../lib/pnl'
import ControlButton from './ControlButton'
import { ExecBar } from './LiveMarketPanel'

/*
  REAL DEMO ACCOUNT (LIVE tab): orders the bot sends to the Exness MT5 DEMO (src/live/execution.py), next to the
  shadow trades. Account, open positions (live P&L ~1 s), pending limit orders, closed trades, stats per strategy,
  P&L curve and daily P&L. Data: /api/live/real (every 5 s) + /api/live/ticks (prices every 1 s).
*/
const POLL_MS = 5_000
const usd = (v: number | null | undefined) =>
  v == null ? '—' : `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
const sgn = (v: number | null | undefined, d = 2) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}`)
const pc = (v: number | null | undefined) => (v == null || v === 0 ? 'var(--text)' : v > 0 ? 'var(--green)' : 'var(--red)')
const dg = (s: string) => (s.endsWith('JPY') ? 3 : 5)
const pip = (s: string) => (s.endsWith('JPY') ? 0.01 : 0.0001)
const nyTime = (sec: number | null | undefined) => (sec ? new Date(sec * 1000).toLocaleString('en-GB', {
  timeZone: 'America/New_York', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) : '—')
const nyIso = (iso: string | null | undefined) => (iso ? nyTime(Date.parse(iso) / 1000) : '—')
const NAME: Record<string, string> = { zone: 'ZONE BOT', orb: 'ORB' }
const STRAT_COLOR: Record<string, string> = { zone: 'var(--cyan)', orb: '#b388ff' }
const EXIT: Record<string, string> = { tp1: '1:1', trail: 'TRAIL', tp2: '1:2' }

function Stat({ label, value, color, tip }: { label: string; value: string; color?: string; tip: string }) {
  return (
    <div title={tip} className="cursor-help">
      <small className="block text-[9px] tracking-wider text-muted">{label}</small>
      <b className="font-display text-[14px] font-medium" style={{ color: color ?? 'var(--text)' }}>{value}</b>
    </div>
  )
}

function Curve({ pts, k }: { pts: RealAccount['curve']; k: 'all' | 'zone' | 'orb' }) {
  if (pts.length < 2) return <div className="flex h-[120px] items-center justify-center text-[10px] text-muted">P&amp;L curve starts after 2 closed real trades.</div>
  const w = 520, h = 120, pad = 6
  const ys = [0, ...pts.map((p) => p[k])]
  const lo = Math.min(...ys), hi = Math.max(...ys)
  const x = (i: number) => pad + (i / (pts.length - 1)) * (w - 2 * pad)
  const y = (v: number) => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad)
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join(' ')
  const last = pts[pts.length - 1][k]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-[120px] w-full" preserveAspectRatio="none">
      <line x1={pad} x2={w - pad} y1={y(0)} y2={y(0)} stroke="var(--line)" strokeDasharray="3 3" />
      <path d={d} fill="none" stroke={last >= 0 ? 'var(--green)' : 'var(--red)'} strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function Daily({ rows, k }: { rows: RealAccount['daily']; k: 'all' | 'zone' | 'orb' }) {
  const vals = rows.map((r) => (k === 'all' ? (r.zone ?? 0) + (r.orb ?? 0) : (r[k] ?? 0)))
  if (!vals.length) return <div className="flex h-[120px] items-center justify-center text-[10px] text-muted">No closed real trades yet.</div>
  const m = Math.max(1, ...vals.map(Math.abs))
  return (
    <div className="flex h-[120px] items-center gap-[2px]">
      {vals.map((v, i) => (
        <div key={rows[i].day} className="flex h-full flex-1 flex-col justify-center" title={`${rows[i].day} (NY): ${usd(v)}`}>
          <div className="flex h-1/2 items-end"><div className="w-full" style={{ height: `${v > 0 ? (v / m) * 100 : 0}%`, background: 'var(--green)' }} /></div>
          <div className="flex h-1/2 items-start"><div className="w-full" style={{ height: `${v < 0 ? (-v / m) * 100 : 0}%`, background: 'var(--red)' }} /></div>
        </div>
      ))}
    </div>
  )
}

export default function RealAccountPanel({ connected }: { connected: boolean }) {
  const [d, setD] = useState<RealAccount | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [f, setF] = useState<'all' | 'zone' | 'orb'>('all')
  useEffect(() => {
    if (!connected) return
    let alive = true
    const load = () => api.liveReal().then((x) => { if (alive) { setD(x); setErr(null) } })
      .catch((e: unknown) => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
    void load()
    const id = window.setInterval(load, POLL_MS)
    return () => { alive = false; window.clearInterval(id) }
  }, [connected])
  const ticks = useLiveTicks(connected && (d?.open?.length ?? 0) > 0)

  const open = useMemo(() => (d?.open ?? []).filter((p) => f === 'all' || p.strategy === f), [d, f])
  const live = (p: RealOpen) => {
    const px = ticks?.prices[p.symbol]
    if (!px) return null
    return floatPnl({ symbol: p.symbol, direction: p.buy ? 'bull' : 'bear', entry: p.price_open, sl: p.sl || p.price_open, lots: p.lots }, px.bid, px.ask)
  }
  const floating = open.reduce((s, p) => s + (live(p)?.usd ?? p.profit) + (p.swap ?? 0), 0)

  const header = (
    <div className="ptitle flex-wrap">
      <span>REAL DEMO ACCOUNT</span>
      <em title="Real orders on the Exness MT5 DEMO account, sent by the bot (zone bot magic 1001, ORB magic 2002). The shadow accounts keep running next to it - compare them here.">
        MT5 DEMO · REAL ORDERS FROM THE BOT · ZONE BOT + ORB · SHADOW KEEPS RUNNING
      </em>
    </div>
  )
  if (!d || d.status === 'unreachable' || d.error) {
    return (
      <div className="panel">{header}
        <div className="py-4 text-center text-[11px] text-muted">{err ? `Could not load: ${err}` : d?.error ?? 'Loading...'}</div>
      </div>
    )
  }
  const ex = d.exec
  const a = ex?.account ?? {}
  const s: RealStats = d.stats?.[f] ?? { trades: 0 }
  const closed: RealClosed[] = (d.closed ?? []).filter((t) => f === 'all' || t.strategy === f)
  const pending = (d.pending ?? []).filter((o) => f === 'all' || o.strategy === f)

  return (
    <div className="panel">
      {header}
      <ExecBar ex={ex} />
      {!ex ? (
        <div className="py-3 text-center text-[11px] text-muted">No real account data yet. It starts with the new bot version on the VPS (git pull + restart FOREX BOT and LIVE API).</div>
      ) : (
        <>
          <div className="mb-2 flex items-center gap-1.5">
            {(['all', 'zone', 'orb'] as const).map((k) => (
              <button key={k} className={`chip ${f === k ? 'on' : ''}`} onClick={() => setF(k)}>{k === 'all' ? 'ALL' : NAME[k]}</button>
            ))}
            <span className="ml-2 text-[10px] text-muted">Account numbers (balance / equity / margin) are for the whole MT5 account; P&amp;L and stats below follow the filter.</span>
          </div>
          <div className="mb-3 grid grid-cols-3 gap-x-3 gap-y-2.5 border p-2.5 sm:grid-cols-4 xl:grid-cols-8" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
            <Stat label="BALANCE" value={usd(a.balance)} color="var(--cyan)" tip={`MT5 ${a.server ?? ''} #${a.login ?? ''} - closed trades only`} />
            <Stat label="EQUITY" value={usd(a.equity)} color={pc((a.equity ?? 0) - (a.balance ?? 0))} tip="Balance + open positions right now (MT5)" />
            <Stat label="FLOATING" value={open.length ? fmtUsd(floating) : '$0.00'} color={pc(floating)} tip="Live profit / loss of the open real positions (filter), after commission + swap, about every second" />
            <Stat label="MARGIN / FREE" value={`${usd((a as { margin?: number }).margin ?? 0)} / ${usd(a.margin_free)}`} tip={`Used margin / free margin${(a as { leverage?: number }).leverage ? ` · leverage 1:${(a as { leverage?: number }).leverage}` : ''}`} />
            <Stat label="NET P&L" value={usd(s.net_usd ?? 0)} color={pc(s.net_usd)} tip="All closed real bot trades (filter), after commission and swap" />
            <Stat label="TODAY / WEEK / MONTH" value={`${sgn(s.today, 0)} / ${sgn(s.week, 0)} / ${sgn(s.month, 0)}`} color={pc(s.today)} tip="Net $ of closed real trades: today, this week, this month (NY time)" />
            <Stat label="TRADES / WIN" value={`${s.trades} / ${s.win_pct ?? '—'}%`} tip={`${s.wins ?? 0} winners`} />
            <Stat label="PF ($)" value={s.pf_usd == null ? '—' : String(s.pf_usd)} color={s.pf_usd != null ? (s.pf_usd >= 1 ? 'var(--green)' : 'var(--red)') : undefined} tip="Profit factor in dollars: gross won / gross lost (after costs)" />
            <Stat label="AVG WIN / LOSS" value={`${usd(s.avg_win)} / ${usd(s.avg_loss)}`} tip="Average net $ of winning / losing real trades" />
            <Stat label="BEST / WORST" value={`${usd(s.best)} / ${usd(s.worst)}`} tip="Best and worst single real trade (net $)" />
            <Stat label="MAX DRAWDOWN" value={usd(-(s.max_dd_usd ?? 0))} color="var(--amber)" tip="Biggest drop of the cumulative net P&L of these trades" />
            <Stat label="COMMISSION / SWAP" value={`${usd(s.commission_usd ?? 0)} / ${usd(s.swap_usd ?? 0)}`} color="var(--red)" tip="Paid to the broker on closed real trades" />
            <Stat label="R REAL / SHADOW" value={`${sgn(s.real_r)} / ${sgn(s.shadow_r)}`} color={pc((s.real_r ?? 0) - (s.shadow_r ?? 0))} tip="Total R of the same trades: real fills vs shadow (green = real did better)" />
            <Stat label="R GAP / TRADE" value={s.r_gap_per_trade == null ? '—' : `${sgn(s.r_gap_per_trade, 3)}R`} color={pc(s.r_gap_per_trade)} tip="Average (real R - shadow R) per trade: what the broker costs you (or gives) per trade" />
            <Stat label="ENTRY SLIP AVG / MAX" value={s.avg_slip_pips == null ? '—' : `${sgn(s.avg_slip_pips, 1)} / ${sgn(s.worst_slip_pips, 1)} p`} color={pc(-(s.avg_slip_pips ?? 0))} tip="Real fill vs shadow fill in pips (+ = worse)" />
            <Stat label="OPEN / PENDING" value={`${open.length} / ${pending.length}`} tip="Real positions open now / zone limit orders waiting on MT5" />
          </div>

          <div className="mb-3 grid gap-3 xl:grid-cols-2">
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">P&amp;L CURVE (CLOSED REAL TRADES, NET $)</div>
              <Curve pts={d.curve ?? []} k={f} />
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">DAILY P&amp;L (LAST 30 DAYS, NY)</div>
              <Daily rows={d.daily ?? []} k={f} />
            </div>
          </div>

          <div className="mb-1 text-[9px] tracking-[2px] text-muted">OPEN REAL POSITIONS ({open.length}) {open.length > 0 && <span className="tracking-[1px]" style={{ color: pc(floating) }}>· {fmtUsd(floating)}</span>}</div>
          {open.length === 0 ? <div className="mb-3 text-[10px] text-muted">No open real positions.</div> : (
            <div className="mb-3 overflow-x-auto">
              <table className="w-full text-[10px]">
                <thead className="text-[9px] text-muted"><tr>
                  <th className="text-left font-normal">STRATEGY</th><th className="text-left font-normal">PAIR</th><th className="text-left font-normal">SIDE</th>
                  <th className="text-left font-normal">OPENED (NY)</th><th className="text-right font-normal">REAL ENTRY</th>
                  <th className="text-right font-normal" title="Entry of the same trade in the shadow">SHADOW</th><th className="text-right font-normal" title="+ = real fill worse">SLIP</th>
                  <th className="text-right font-normal">SL</th><th className="text-right font-normal">TP</th><th className="text-right font-normal">EXIT</th>
                  <th className="text-right font-normal">NOW</th><th className="text-right font-normal">PIPS</th><th className="text-right font-normal">P&amp;L</th>
                  <th className="text-right font-normal">R</th><th className="text-right font-normal">SWAP</th><th className="text-right font-normal">TICKET</th><th />
                </tr></thead>
                <tbody>{open.map((p) => {
                  const l = live(p)
                  const px = ticks?.prices[p.symbol]
                  const now = px ? (p.buy ? px.bid : px.ask) : p.price_now
                  const r = l && p.risk ? (l.pips * pip(p.symbol)) / p.risk : p.r_now
                  return (
                    <tr key={p.ticket} className="border-t" style={{ borderColor: 'var(--line)' }}>
                      <td style={{ color: STRAT_COLOR[p.strategy] }}>{NAME[p.strategy]}{p.taken_manual ? ' (TAKE)' : ''}</td>
                      <td>{p.symbol}</td>
                      <td style={{ color: p.buy ? 'var(--green)' : 'var(--red)' }}>{p.buy ? 'BUY' : 'SELL'} {p.lots}</td>
                      <td className="text-muted">{p.opened ? nyIso(p.opened) : nyTime(p.time)}</td>
                      <td className="text-right">{p.price_open.toFixed(dg(p.symbol))}</td>
                      <td className="text-right text-muted">{p.shadow_entry != null ? p.shadow_entry.toFixed(dg(p.symbol)) : '—'}</td>
                      <td className="text-right" style={{ color: (p.slip_pips ?? 0) > 0 ? 'var(--red)' : undefined }}>{p.slip_pips == null ? '—' : sgn(p.slip_pips, 1)}</td>
                      <td className="text-right" style={{ color: 'var(--red)' }}>{p.sl ? p.sl.toFixed(dg(p.symbol)) : '—'}</td>
                      <td className="text-right" style={{ color: 'var(--green)' }}>{p.tp ? p.tp.toFixed(dg(p.symbol)) : '—'}</td>
                      <td className="text-right" style={{ color: p.exit === 'trail' ? '#ffb020' : 'var(--green)' }}>{EXIT[p.exit ?? ''] ?? '—'}{p.user_exit ? '*' : ''}</td>
                      <td className="text-right">{now != null ? now.toFixed(dg(p.symbol)) : '—'}</td>
                      <td className="text-right" style={{ color: pc(l?.pips) }}>{l ? fmtSigned(l.pips) : '—'}</td>
                      <td className="text-right" style={{ color: pc(l?.usd ?? p.profit) }}>{l ? fmtUsd(l.usd) : usd(p.profit)}</td>
                      <td className="text-right" style={{ color: pc(r) }}>{r == null ? '—' : `${sgn(r)}R`}</td>
                      <td className="text-right text-muted">{usd(p.swap)}</td>
                      <td className="text-right text-muted">#{p.ticket}</td>
                      <td className="pl-1 text-right">
                        {p.shadow_fill_time != null && (
                          <ControlButton
                            className="chip !px-1.5 !py-0 text-[9px]" style={{ color: 'var(--red)', borderColor: 'var(--red)' }}
                            title="Close this trade now (shadow + real) at the market price"
                            control={{ action: p.strategy === 'orb' ? 'orb_close' : 'close', symbol: p.symbol, fill_time: p.shadow_fill_time }}
                            question={`Close this ${NAME[p.strategy]} trade now?\n${p.symbol} ${p.buy ? 'BUY' : 'SELL'} ${p.lots} lots · real entry ${p.price_open}${l ? ` · now ${fmtUsd(l.usd)}` : ''}\nThe shadow trade and the real MT5 position both close. This cannot be undone.`}
                            confirmText="CLOSE NOW"
                          >CLOSE</ControlButton>
                        )}
                      </td>
                    </tr>
                  )
                })}</tbody>
              </table>
            </div>
          )}

          <div className="mb-1 text-[9px] tracking-[2px] text-muted">PENDING LIMIT ORDERS ON MT5 ({pending.length})</div>
          {pending.length === 0 ? <div className="mb-3 text-[10px] text-muted">No pending orders (zone bot limit orders show here while a zone is READY).</div> : (
            <table className="mb-3 w-full text-[10px]">
              <thead className="text-[9px] text-muted"><tr>
                <th className="text-left font-normal">STRATEGY</th><th className="text-left font-normal">PAIR</th><th className="text-left font-normal">SIDE</th>
                <th className="text-right font-normal">LIMIT</th><th className="text-right font-normal">SL</th><th className="text-right font-normal">TP</th>
                <th className="text-right font-normal">LOTS</th><th className="text-right font-normal">PLACED (NY)</th><th className="text-right font-normal">TICKET</th>
              </tr></thead>
              <tbody>{pending.map((o) => (
                <tr key={o.ticket} className="border-t" style={{ borderColor: 'var(--line)' }}>
                  <td style={{ color: STRAT_COLOR[o.strategy] }}>{NAME[o.strategy]}</td><td>{o.symbol}</td>
                  <td style={{ color: (o as { buy?: boolean }).buy ? 'var(--green)' : 'var(--red)' }}>{(o as { buy?: boolean }).buy ? 'BUY LIMIT' : 'SELL LIMIT'}</td>
                  <td className="text-right">{o.price.toFixed(dg(o.symbol))}</td>
                  <td className="text-right" style={{ color: 'var(--red)' }}>{o.sl ? o.sl.toFixed(dg(o.symbol)) : '—'}</td>
                  <td className="text-right" style={{ color: 'var(--green)' }}>{o.tp ? o.tp.toFixed(dg(o.symbol)) : 'TRAIL'}</td>
                  <td className="text-right">{o.lots}</td>
                  <td className="text-right text-muted">{nyTime((o as { time?: number }).time)}</td>
                  <td className="text-right text-muted">#{o.ticket}</td>
                </tr>
              ))}</tbody>
            </table>
          )}

          <div className="mb-1 text-[9px] tracking-[2px] text-muted">CLOSED REAL TRADES ({closed.length})</div>
          {closed.length === 0 ? <div className="mb-3 text-[10px] text-muted">No closed real trades yet.</div> : (
            <div className="mb-3 max-h-[40vh] overflow-auto">
              <table className="w-full text-[10px]">
                <thead className="sticky top-0 text-[9px] text-muted" style={{ background: 'var(--panel)' }}><tr>
                  <th className="text-left font-normal">CLOSED (NY)</th><th className="text-left font-normal">STRATEGY</th><th className="text-left font-normal">PAIR</th>
                  <th className="text-left font-normal">SIDE</th><th className="text-right font-normal">SHADOW ENTRY</th><th className="text-right font-normal">REAL ENTRY</th>
                  <th className="text-right font-normal">SLIP</th><th className="text-right font-normal">REAL EXIT</th><th className="text-right font-normal">WHY</th>
                  <th className="text-right font-normal">SHADOW R</th><th className="text-right font-normal">REAL R</th><th className="text-right font-normal">GROSS</th>
                  <th className="text-right font-normal">COMM.</th><th className="text-right font-normal">SWAP</th><th className="text-right font-normal">NET</th>
                </tr></thead>
                <tbody>{closed.map((t, i) => (
                  <tr key={`${t.ticket}${i}`} className="border-t" style={{ borderColor: 'var(--line)' }} title={`${t.note ?? ''}${t.real_open_time ? ` · opened ${nyIso(t.real_open_time)} NY` : ''} · ticket #${t.ticket ?? '?'}`}>
                    <td className="text-muted">{nyIso(t.real_exit_time)}</td>
                    <td style={{ color: STRAT_COLOR[t.strategy] }}>{NAME[t.strategy] ?? t.strategy}</td>
                    <td>{t.symbol}</td>
                    <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'} {t.lots ?? ''}</td>
                    <td className="text-right text-muted">{t.shadow_entry ?? '—'}</td>
                    <td className="text-right">{t.real_entry ?? '—'}</td>
                    <td className="text-right" style={{ color: (t.entry_slip_pips ?? 0) > 0 ? 'var(--red)' : undefined }}>{t.entry_slip_pips == null ? '—' : sgn(t.entry_slip_pips, 1)}</td>
                    <td className="text-right">{t.real_exit ?? '—'}</td>
                    <td className="text-right text-muted" title={t.shadow_exit ? `shadow exit: ${t.shadow_exit}` : ''}>{t.real_exit_reason ?? '—'}</td>
                    <td className="text-right" style={{ color: pc(t.shadow_r) }}>{sgn(t.shadow_r)}</td>
                    <td className="text-right" style={{ color: pc(t.real_r) }}>{sgn(t.real_r)}</td>
                    <td className="text-right" style={{ color: pc(t.profit_usd) }}>{usd(t.profit_usd)}</td>
                    <td className="text-right" style={{ color: 'var(--red)' }}>{usd(t.commission_usd)}</td>
                    <td className="text-right text-muted">{usd(t.swap_usd)}</td>
                    <td className="text-right font-medium" style={{ color: pc(t.net_usd) }}>{usd(t.net_usd)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}

          {(ex.errors?.length ?? 0) > 0 && (
            <>
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">LAST ORDER PROBLEMS</div>
              <div className="text-[10px]" style={{ color: 'var(--amber)' }}>{ex.errors.slice().reverse().map((e, i) => <div key={i}>{e}</div>)}</div>
            </>
          )}
        </>
      )}
    </div>
  )
}
