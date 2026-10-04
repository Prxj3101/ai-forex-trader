import { AreaSeries, ColorType, createChart, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from 'lightweight-charts'
import { useEffect, useRef, useState } from 'react'
import { api, type LiveStatus } from '../lib/api'
import { rgba, useTheme } from '../lib/theme'
import { fmtTs } from '../lib/time'

/*
  LIVE SHADOW ACCOUNT: shadow bot (src/live/shadow.py) jo files likhta hai, unka live view.
  Har 15 second /api/live poll. Koi asli order nahi - sab paper trading.
*/

const POLL_MS = 15_000
const usd = (v: number) => `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
const signed = (v: number, d = 2) => `${v > 0 ? '+' : ''}${v.toFixed(d)}`
const pnlColor = (v: number) => (v > 0 ? 'var(--green)' : v < 0 ? 'var(--red)' : 'var(--text)')

const DECISION_LABEL: Record<string, { text: string; tip: string }> = {
  taken: { text: 'TAKEN', tip: 'The bot opened this trade' },
  fill_skipped: { text: 'FILL SKIPPED', tip: 'The bot wanted it, but a limit or daily rule blocked it at the fill' },
  ml_low: { text: 'ML LOW', tip: 'ML score below the top-30% threshold' },
  value_area: { text: 'VALUE AREA', tip: 'Entry on the expensive side of the last 24h value area' },
  quiet: { text: 'QUIET MARKET', tip: 'Tick volume before the touch was below 0.9x the 24h average' },
  news: { text: 'NEWS BLOCK', tip: 'Inside the window around high-impact news' },
  limits: { text: 'LIMITS', tip: 'Max open trades, already in this pair, or too much USD one way' },
  daily_limit: { text: 'DAILY RULE', tip: 'Daily stop or daily risk budget' },
  kill_switch: { text: 'KILL SWITCH', tip: 'Kill switch was active' },
  lot_small: { text: 'LOT TOO SMALL', tip: 'Lot size below 0.01' },
}

function decisionColor(d: string) {
  if (d === 'FILLED' || d === 'TAKE') return 'var(--green)'
  if (d === 'FILL-SKIPPED') return 'var(--amber)'
  return 'var(--muted)'
}

function MiniCurve({ data, start }: { data: { time: number; value: number }[]; start: number }) {
  const { theme } = useTheme()
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<ISeriesApi<'Area'> | null>(null)

  useEffect(() => {
    if (!box.current) return
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#c3cfdf', fontFamily: "'Share Tech Mono', monospace", attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(255,255,255,0.06)' } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number), priceFormatter: (p: number) => `$${Math.round(p).toLocaleString()}` },
      handleScroll: false,
      handleScale: false,
    })
    series.current = c.addSeries(AreaSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true })
    chart.current = c
    return () => c.remove()
  }, [])

  useEffect(() => {
    series.current?.applyOptions({ lineColor: theme.accent, topColor: rgba(theme, 0.3), bottomColor: rgba(theme, 0.02) })
  }, [theme])

  useEffect(() => {
    // start point so a single trade still draws a line; times must be strictly increasing
    const pts: { time: UTCTimestamp; value: number }[] = []
    const first = data.length ? data[0].time - 300 : Math.floor(Date.now() / 1000)
    pts.push({ time: first as UTCTimestamp, value: start })
    for (const p of data) {
      const t = Math.max(p.time, (pts[pts.length - 1].time as number) + 1)
      pts.push({ time: t as UTCTimestamp, value: p.value })
    }
    series.current?.setData(pts)
    chart.current?.timeScale().fitContent()
  }, [data, start])

  return <div ref={box} className="h-[150px] w-full" />
}

function Stat({ label, value, color, tip }: { label: string; value: string; color?: string; tip: string }) {
  return (
    <div title={tip} className="cursor-help">
      <small className="block text-[9px] tracking-wider text-muted">{label}</small>
      <b className="font-display text-[15px] font-medium" style={{ color: color ?? 'var(--text)' }}>{value}</b>
    </div>
  )
}

function ago(sec: number | null | undefined) {
  if (sec == null) return '?'
  if (sec < 90) return `${sec}s ago`
  if (sec < 5400) return `${Math.round(sec / 60)} min ago`
  if (sec < 172800) return `${Math.round(sec / 3600)} h ago`
  return `${Math.round(sec / 86400)} days ago`
}

export default function LivePanel({ connected }: { connected: boolean }) {
  const [data, setData] = useState<LiveStatus | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)                 // REFRESH button dabane pe badhta hai -> turant reload
  const [loading, setLoading] = useState(false)
  const [loadedAt, setLoadedAt] = useState<Date | null>(null)

  useEffect(() => {
    if (!connected) return
    let alive = true
    const load = () => {
      setLoading(true)
      api.live()
        .then((d) => { if (alive) { setData(d); setErr(null); setLoadedAt(new Date()) } })
        .catch((e: unknown) => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
        .finally(() => { if (alive) setLoading(false) })
    }
    load()
    const id = window.setInterval(load, POLL_MS)     // har 15 s apne aap bhi
    return () => { alive = false; window.clearInterval(id) }
  }, [connected, tick])

  const running = data?.status === 'running'
  const statusColor = running
    ? 'var(--green)'
    : data?.status === 'stopped'
      ? 'var(--amber)'
      : data?.status === 'unreachable'
        ? 'var(--red)'
        : 'var(--muted)'
  const unreachable = data?.status === 'unreachable'
  const statusText = !data
    ? 'LOADING'
    : unreachable
      ? 'VPS NOT REACHABLE'
      : data.status === 'none'
        ? 'NOT STARTED'
        : running
          ? `RUNNING · BOT LAST CHECK ${ago(data.age_sec).toUpperCase()}`
          : `STOPPED · BOT LAST CHECK ${ago(data.age_sec).toUpperCase()}`
  const sourceText = data?.source === 'vps' ? 'VPS' : data?.source === 'local' ? 'THIS PC' : null

  const header = (
    <div className="ptitle flex-wrap">
      <span>LIVE SHADOW ACCOUNT</span>
      <em className="flex items-center gap-2" title="Paper trading on live MT5 prices. The bot sends no real orders.">
        <i className="inline-block h-2 w-2 rounded-full" style={{ background: statusColor, boxShadow: running ? `0 0 8px ${statusColor}` : 'none' }} />
        <span style={{ color: statusColor }} title="The bot saves its account once per new 5-minute candle, so 0 to 5 minutes is normal. Over 10 minutes means it has stopped.">{statusText}</span>
        {sourceText && (
          <span title={sourceText === 'VPS' ? 'Data comes from the bot running on the VPS (read-only).' : 'Data comes from the bot folder on this computer.'}>
            · SOURCE {sourceText}
          </span>
        )}
        · PAPER TRADING · NO REAL ORDERS
        <button
          className="chip"
          disabled={!connected || loading}
          onClick={() => setTick((t) => t + 1)}
          title={`Reload now. It also refreshes by itself every ${POLL_MS / 1000} seconds.${loadedAt ? ` Last loaded ${loadedAt.toLocaleTimeString('en-GB', { hour12: false })}.` : ''}`}
        >
          {loading ? 'REFRESHING...' : '⟳ REFRESH'}
        </button>
        {loadedAt && (
          <span title="When this panel last loaded data from the backend">PANEL {loadedAt.toLocaleTimeString('en-GB', { hour12: false })}</span>
        )}
      </em>
    </div>
  )

  if (!data || data.status === 'none' || !data.account) {
    return (
      <div className="panel">
        {header}
        <div className="py-6 text-center text-[11px] text-muted">
          {err
            ? `Could not load live data: ${err}`
            : unreachable
              ? `${data?.error ?? 'Cannot reach the VPS.'} Check that the VPS is running, live_api is started, and your home IP is allowed in the AWS security group.`
              : 'No live shadow data yet. Start the bot: python src\\live\\shadow.py (MT5 demo must be open).'}
        </div>
      </div>
    )
  }

  const a = data.account
  const c = data.closed ?? { count: 0, wins: 0, win_pct: null, pnl: 0, r_sum: 0, pf: null }
  const ddColor = a.dd_pct >= a.kill_dd_pct ? 'var(--red)' : a.dd_pct >= a.brake_dd_pct ? 'var(--amber)' : 'var(--text)'
  const open = data.open_trades ?? []
  const trades = data.trades ?? []
  const decisions = data.decisions ?? []
  const whatif = data.whatif ?? []

  return (
    <div className="panel">
      {header}

      {!data.full_state && (
        <div className="mb-2 text-[10px]" style={{ color: 'var(--amber)' }}>
          Old state file format: account history restarts when the bot is restarted with the new version.
        </div>
      )}

      <div className="mb-3 grid grid-cols-3 gap-x-3 gap-y-2.5 border p-2.5 sm:grid-cols-5 xl:grid-cols-9" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
        <Stat label="EQUITY" value={usd(a.equity)} color="var(--cyan)" tip={`Started with ${usd(a.capital)} · closed trades only (open trades not counted until they close)`} />
        <Stat label="TOTAL" value={`${signed(a.total_pct)}%`} color={pnlColor(a.total_pct)} tip="Change since the start of the shadow account" />
        <Stat label="TODAY" value={usd(a.today_pnl)} color={pnlColor(a.today_pnl)} tip={`Broker day (5 PM to 5 PM New York). Daily stop at -${a.daily_loss_pct}%, daily risk budget ${a.daily_budget_pct}%`} />
        <Stat label="THIS MONTH" value={usd(a.month_pnl)} color={pnlColor(a.month_pnl)} tip="Profit this month. Feeds the house money risk boost." />
        <Stat label="DRAWDOWN" value={`-${a.dd_pct.toFixed(2)}%`} color={ddColor} tip={`From the peak ${usd(a.peak)}. Brake at -${a.brake_dd_pct}%, kill switch at -${a.kill_dd_pct}%`} />
        <Stat
          label="RISK NOW"
          value={a.kill ? 'STOPPED' : `${a.risk_now_pct.toFixed(2)}%`}
          color={a.kill ? 'var(--red)' : a.braked ? 'var(--amber)' : 'var(--text)'}
          tip={a.kill ? 'Kill switch active: no new trades until reviewed' : a.braked ? 'Drawdown brake active: risk cut in half' : 'Risk per trade for the next trade (0.25% + house money, max 0.5%)'}
        />
        <Stat label="TRADES" value={`${c.count}`} tip={`${c.wins} winners`} />
        <Stat label="WIN RATE" value={c.win_pct == null ? '—' : `${c.win_pct}%`} tip="Closed trades with profit after costs" />
        <Stat label="PF / R" value={`${c.pf == null ? '—' : c.pf.toFixed(2)} / ${signed(c.r_sum, 1)}R`} color={pnlColor(c.r_sum)} tip="Profit factor (money won / money lost) and total R of closed trades" />
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,0.9fr)]">
        {/* trades */}
        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">OPEN TRADES ({open.length})</div>
          {open.length === 0 ? (
            <div className="mb-3 text-[10px] text-muted">No open trades.</div>
          ) : (
            <table className="mb-3 w-full text-[10px]">
              <thead className="text-muted"><tr><th className="text-left font-normal">PAIR</th><th className="text-left font-normal">SIDE</th><th className="text-right font-normal">ENTRY</th><th className="text-right font-normal">SL PIPS</th><th className="text-right font-normal">LOTS</th><th className="text-right font-normal">EXIT</th><th className="text-right font-normal">R NOW</th></tr></thead>
              <tbody>
                {open.map((t, i) => (
                  <tr key={i} title={t.fill_time ? `Filled ${fmtTs(t.fill_time)} NY · ML score ${(t.score * 100).toFixed(0)}% · risk $${t.risk_usd}` : undefined}>
                    <td>{t.symbol}</td>
                    <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'}</td>
                    <td className="text-right">{t.entry}</td>
                    <td className="text-right">{t.sl_pips}</td>
                    <td className="text-right">{t.lots}</td>
                    <td className="text-right">{t.exit === 'trail' ? 'TRAIL' : '1:1'}</td>
                    <td className="text-right" style={{ color: pnlColor(t.r_now ?? 0) }}>{t.r_now == null ? '—' : signed(t.r_now)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="mb-1 text-[9px] tracking-[2px] text-muted">CLOSED TRADES (LATEST {trades.length})</div>
          {trades.length === 0 ? (
            <div className="text-[10px] text-muted">No closed trades yet.</div>
          ) : (
            <div className="max-h-[210px] overflow-y-auto">
              <table className="w-full text-[10px]">
                <thead className="sticky top-0 text-muted" style={{ background: 'var(--panel)' }}><tr><th className="text-left font-normal">CLOSED (NY)</th><th className="text-left font-normal">PAIR</th><th className="text-left font-normal">SIDE</th><th className="text-right font-normal">EXIT</th><th className="text-right font-normal">R</th><th className="text-right font-normal">P&amp;L</th></tr></thead>
                <tbody>
                  {trades.map((t, i) => (
                    <tr key={i} title={`Filled ${t.fill_ts ? fmtTs(t.fill_ts) : '?'} NY at ${t.entry} · ${t.lots} lots · closed by ${t.exit_reason}`}>
                      <td className="text-muted">{t.exit_ts ? fmtTs(t.exit_ts) : '—'}</td>
                      <td>{t.symbol}</td>
                      <td style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'}</td>
                      <td className="text-right">{t.exit === 'trail' ? 'TRAIL' : '1:1'} · {t.exit_reason?.toUpperCase()}</td>
                      <td className="text-right" style={{ color: pnlColor(t.r) }}>{signed(t.r)}</td>
                      <td className="text-right" style={{ color: pnlColor(t.pnl) }}>{usd(t.pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* decisions */}
        <div className="min-w-0">
          <div className="mb-1.5 text-[10px] tracking-[2px] text-muted">LATEST DECISIONS (NY TIME)</div>
          <div className="flex max-h-[360px] flex-col gap-1.5 overflow-y-auto pr-1">
            {decisions.length === 0 && <div className="text-[12px] text-muted">No decisions yet.</div>}
            {decisions.map((d, i) => {
              const col = decisionColor(d.decision)
              return (
                <div key={i} className="border-l-2 py-1 pl-2" style={{ borderColor: col, background: 'var(--panel2)' }}>
                  <div className="flex flex-wrap items-baseline gap-x-2 text-[12px]">
                    <span className="text-muted">{d.bar_ts ? fmtTs(d.bar_ts) : ''}</span>
                    <b className="font-medium">{d.symbol}</b>
                    <span style={{ color: d.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{d.direction === 'bull' ? 'BUY' : 'SELL'}</span>
                    <span>@ {d.entry}</span>
                    <span className="text-muted">ML {Math.round(d.score * 100)}%</span>
                    <b className="ml-auto font-display text-[11px] font-medium tracking-wider" style={{ color: col }}>{d.decision}</b>
                  </div>
                  {d.reason && <div className="mt-0.5 text-[11px] leading-snug text-muted">{d.reason}</div>}
                </div>
              )
            })}
          </div>
        </div>

        {/* curve + what-if */}
        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">BALANCE (CLOSED TRADES)</div>
          <MiniCurve data={data.curve ?? []} start={a.capital} />
          <div className="mb-1 mt-2 text-[9px] tracking-[2px] text-muted" title="Every zone price touched, what the bot decided, and the result if that trade had been taken (net R after costs)">
            WHAT IF (ZONES TOUCHED, NET R)
          </div>
          {whatif.length === 0 ? (
            <div className="text-[10px] text-muted">Fills when price touches zones. Needs the new shadow version.</div>
          ) : (
            <table className="w-full text-[10px]">
              <thead className="text-muted"><tr><th className="text-left font-normal">DECISION</th><th className="text-right font-normal">ZONES</th><th className="text-right font-normal">WIN</th><th className="text-right font-normal" title="Exit chosen by the exit brain">BRAIN</th><th className="text-right font-normal">1:1</th><th className="text-right font-normal">TRAIL</th></tr></thead>
              <tbody>
                {whatif.map((w) => (
                  <tr key={w.decision} title={DECISION_LABEL[w.decision]?.tip ?? w.decision}>
                    <td style={{ color: w.decision === 'taken' ? 'var(--green)' : 'var(--text)' }}>{DECISION_LABEL[w.decision]?.text ?? w.decision.toUpperCase()}</td>
                    <td className="text-right">{w.zones}</td>
                    <td className="text-right">{w.win_pct}%</td>
                    <td className="text-right" style={{ color: pnlColor(w.net_brain) }}>{signed(w.net_brain, 1)}</td>
                    <td className="text-right" style={{ color: pnlColor(w.net_tp1) }}>{signed(w.net_tp1, 1)}</td>
                    <td className="text-right" style={{ color: pnlColor(w.net_trail) }}>{signed(w.net_trail, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="mt-1 text-[9px] text-muted">Skipped rows should end up worse than TAKEN. Judge only after 100+ trades.</div>
        </div>
      </div>
    </div>
  )
}
