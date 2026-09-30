import { ColorType, LineSeries, createChart, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from 'lightweight-charts'
import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { api, type BreakdownRow, type ResearchFilters, type ResearchReport, type Stats } from '../lib/api'
import { rgba, useTheme } from '../lib/theme'
import { fmtTs } from '../lib/time'

/*
  Strategy research: Setup #1 (orderflow zone tap) ke 10 saal ke results.
  Left: filters ("what if main sirf aise zones leta?")
  Middle: ALL zones vs YOUR FILTER ka comparison + equity curve
  Bottom: session / displacement / year / direction ke hisaab se breakdown
*/

const SESSIONS = ['Asian', 'London', 'London+NY', 'New York', 'Off-hours']
const DEFAULT_FILTERS: ResearchFilters = { min_disp_h1: 0, sessions: [], fvg_only: false, confluence_only: false, direction: 'both' }

type Props = { symbols: Record<string, string[]>; defaultSymbol: string; refreshKey: number; building: boolean }

const pfColor = (pf: number | null) => (pf === null ? 'var(--muted)' : pf >= 1.05 ? 'var(--green)' : pf < 0.98 ? 'var(--red)' : 'var(--amber)')

function StatBlock({ title, s, color, beWin }: { title: string; s: Stats; color: string; beWin: number }) {
  const cells: [string, string, string, string][] = [
    ['TRADES', s.trades.toLocaleString(), 'var(--text)', 'Zones that got filled (limit order touched)'],
    ['WIN RATE', s.win_pct === null ? '--' : `${s.win_pct}%`, s.win_pct !== null && s.win_pct > beWin ? 'var(--green)' : 'var(--red)', `Break-even at 1:2 is ${beWin}%`],
    ['PROFIT FACTOR', s.pf === null ? '--' : s.pf.toFixed(2), pfColor(s.pf), 'Gross win R / gross loss R. Above 1 = profitable'],
    ['TOTAL R', `${s.total_r > 0 ? '+' : ''}${s.total_r}R`, s.total_r >= 0 ? 'var(--green)' : 'var(--red)', 'Sum of all trade results in risk units (1R = one stop loss)'],
    ['MAX DRAWDOWN', `${s.max_dd_r ?? 0}R`, 'var(--red)', 'Largest peak-to-trough drop of cumulative R'],
  ]
  return (
    <div className="border p-2.5" style={{ borderColor: color, background: 'var(--panel2)' }}>
      <div className="font-display mb-2 text-[10px] tracking-[2px]" style={{ color }}>{title}</div>
      <div className="grid grid-cols-5 gap-2">
        {cells.map(([label, value, c, tip]) => (
          <div key={label} title={tip} className="cursor-help">
            <small className="block text-[9px] tracking-wider text-muted">{label}</small>
            <b className="font-display text-[15px] font-medium" style={{ color: c }}>{value}</b>
          </div>
        ))}
      </div>
    </div>
  )
}

function Breakdown({ title, rows }: { title: string; rows: BreakdownRow[] }) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] tracking-[2px] text-muted">{title}</div>
      {rows.length === 0 && <div className="text-[11px] text-muted">No trades</div>}
      {rows.map((r) => {
        const pf = r.pf ?? 0
        const width = Math.min(pf / 2, 1) * 100
        return (
          <div key={r.label} className="mb-1 grid grid-cols-[72px_1fr_44px] items-center gap-2 text-[11px]" title={`${r.trades} trades, win ${r.win_pct ?? '--'}%, ${r.total_r}R`}>
            <span className="truncate">{r.label}</span>
            <div className="relative h-2.5" style={{ background: 'rgba(255,255,255,0.05)' }}>
              <motion.i
                className="absolute inset-y-0 left-0 block"
                initial={{ width: 0 }}
                animate={{ width: `${width}%` }}
                transition={{ duration: 0.6 }}
                style={{ background: pfColor(r.pf), boxShadow: `0 0 8px ${pfColor(r.pf)}` }}
              />
              <span className="absolute inset-y-[-3px] left-1/2 w-px" style={{ background: 'var(--text)', opacity: 0.6 }} title="PF 1.0 = break-even" />
            </div>
            <b className="text-right font-normal" style={{ color: pfColor(r.pf) }}>{r.pf?.toFixed(2) ?? '--'}</b>
          </div>
        )
      })}
    </div>
  )
}

function EquityChart({ report }: { report: ResearchReport }) {
  const { theme } = useTheme()
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const allS = useRef<ISeriesApi<'Line'> | null>(null)
  const fltS = useRef<ISeriesApi<'Line'> | null>(null)

  useEffect(() => {
    if (!box.current) return
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#c3cfdf', fontFamily: "'Share Tech Mono', monospace", attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(255,255,255,0.06)' } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, tickMarkFormatter: (t: Time) => new Date((t as number) * 1000).getUTCFullYear().toString() },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number), priceFormatter: (p: number) => `${p.toFixed(0)}R` },
      handleScroll: false,
      handleScale: false,
    })
    allS.current = c.addSeries(LineSeries, { color: '#9aa8bd', lineWidth: 1, priceLineVisible: false, lastValueVisible: true, title: 'ALL' })
    fltS.current = c.addSeries(LineSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true, title: 'FILTER' })
    chart.current = c
    return () => c.remove()
  }, [])

  useEffect(() => {
    fltS.current?.applyOptions({ color: theme.accent })
    chart.current?.applyOptions({ crosshair: { vertLine: { color: rgba(theme, 0.5) }, horzLine: { color: rgba(theme, 0.5) } } })
  }, [theme])

  useEffect(() => {
    allS.current?.setData(report.equity_all.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })))
    fltS.current?.setData(report.equity_filtered.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })))
    chart.current?.timeScale().fitContent()
  }, [report])

  return <div ref={box} className="h-[220px] w-full" />
}

export default function ResearchPanel({ symbols, defaultSymbol, refreshKey, building }: Props) {
  const [symbol, setSymbol] = useState(defaultSymbol)
  const [tf, setTf] = useState('M15')
  const [filters, setFilters] = useState<ResearchFilters>(DEFAULT_FILTERS)
  const [result, setResult] = useState<{ key: string; report: ResearchReport | null; error: string | null } | null>(null)
  const [buildError, setBuildError] = useState<string | null>(null)

  const key = JSON.stringify({ symbol, tf, filters, refreshKey })

  useEffect(() => {
    let cancelled = false
    api.researchReport(symbol, tf, filters)
      .then((report) => !cancelled && setResult({ key, report, error: null }))
      .catch((e: unknown) => !cancelled && setResult({ key, report: null, error: e instanceof Error ? e.message : String(e) }))
    return () => {
      cancelled = true
    }
  }, [symbol, tf, filters, refreshKey, key])

  const report = result?.report ?? null
  const error = result?.key === key ? result.error : null
  const set = <K extends keyof ResearchFilters>(k: K, v: ResearchFilters[K]) => setFilters((f) => ({ ...f, [k]: v }))
  const toggleSession = (s: string) =>
    set('sessions', filters.sessions.includes(s) ? filters.sessions.filter((x) => x !== s) : [...filters.sessions, s])

  const build = async () => {
    setBuildError(null)
    try {
      await api.researchBuild({ symbol, tf, swing_n: 1, min_disp: 2 })
    } catch (e) {
      setBuildError(e instanceof Error ? e.message : String(e))
    }
  }

  const be = report?.break_even_win_pct ?? 33.3

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>STRATEGY RESEARCH // SETUP #1 ORDERFLOW ZONE TAP</span>
        <em>10 YEARS · REAL SPREAD · 1:2 RR · LIMIT AT ZONE BODY · SL BEYOND WICK</em>
      </div>

      <div className="grid gap-4 xl:grid-cols-[250px_minmax(0,1fr)]">
        {/* ---------------- Filters ---------------- */}
        <div className="flex flex-col gap-3 text-[11px]">
          <div className="flex flex-wrap items-center gap-2">
            <select className="field" value={symbol} onChange={(e) => setSymbol(e.target.value)}>
              {Object.keys(symbols).map((s) => <option key={s}>{s}</option>)}
            </select>
            {['M15', 'H1'].map((t) => (
              <button key={t} className={`chip ${tf === t ? 'on' : ''}`} onClick={() => setTf(t)}>{t}</button>
            ))}
          </div>
          <button className="btn" onClick={build} disabled={building} title="Recompute every zone and simulate every trade on the full history">
            {building ? 'BUILDING...' : report ? 'REBUILD DATASET' : 'BUILD DATASET'}
          </button>
          {buildError && <div style={{ color: 'var(--red)' }}>{buildError}</div>}

          <div className="mt-1 text-[10px] tracking-[2px] text-muted">FILTERS (WHAT IF)</div>
          <label className="flex flex-col gap-1" title="Displacement measured in H1 ATR. Bigger = stronger move on the higher timeframe">
            <span className="flex justify-between"><span>MIN H1 DISPLACEMENT</span><b className="font-normal" style={{ color: 'var(--cyan)' }}>{filters.min_disp_h1.toFixed(2)}x</b></span>
            <input type="range" min={0} max={3} step={0.25} value={filters.min_disp_h1} onChange={(e) => set('min_disp_h1', Number(e.target.value))} />
          </label>
          <div title="Session when the limit order got filled. None selected = all sessions">
            <div className="mb-1">SESSION</div>
            <div className="flex flex-wrap gap-1">
              {SESSIONS.map((s) => (
                <button key={s} className={`chip ${filters.sessions.includes(s) ? 'on' : ''}`} onClick={() => toggleSession(s)}>{s}</button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            <button className={`chip ${filters.fvg_only ? 'on' : ''}`} onClick={() => set('fvg_only', !filters.fvg_only)} title="Only zones whose displacement left a fair value gap">FVG ONLY</button>
            <button className={`chip ${filters.confluence_only ? 'on' : ''}`} onClick={() => set('confluence_only', !filters.confluence_only)} title="Only zones that overlap a live H1 orderflow zone in the same direction">H1 CONFLUENCE</button>
          </div>
          <div className="flex flex-wrap gap-1">
            {(['both', 'bull', 'bear'] as const).map((d) => (
              <button key={d} className={`chip ${filters.direction === d ? 'on' : ''}`} onClick={() => set('direction', d)}>
                {d === 'both' ? 'BUY + SELL' : d === 'bull' ? 'BUY ONLY' : 'SELL ONLY'}
              </button>
            ))}
          </div>
          <button className="chip self-start" onClick={() => setFilters(DEFAULT_FILTERS)}>RESET FILTERS</button>
        </div>

        {/* ---------------- Results ---------------- */}
        <div className="min-w-0">
          {!report && (
            <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-2 text-center text-muted">
              <div className="font-display tracking-[3px]" style={{ color: 'var(--amber)' }}>NO DATASET YET</div>
              <div>{error ?? 'Build the dataset to simulate every orderflow zone on the full history.'}</div>
            </div>
          )}
          {report && (
            <>
              <div className="mb-2 grid gap-2 2xl:grid-cols-2">
                <StatBlock title="ALL ZONES (NO FILTER)" s={report.all} color="#9aa8bd" beWin={be} />
                <StatBlock title="YOUR FILTER" s={report.filtered} color="var(--cyan)" beWin={be} />
              </div>
              <div className="mb-1 flex flex-wrap justify-between text-[10px] tracking-wider text-muted">
                <span>EQUITY CURVE (CUMULATIVE R, 1R = ONE STOP LOSS)</span>
                <span>BREAK-EVEN WIN RATE AT 1:2 = {be}%</span>
              </div>
              <EquityChart report={report} />
              <div className="mt-3 grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
                <Breakdown title="BY SESSION (PF)" rows={report.by_session} />
                <Breakdown title="BY H1 DISPLACEMENT (PF)" rows={report.by_disp} />
                <Breakdown title="BY DIRECTION (PF)" rows={report.by_direction} />
                <Breakdown title="BY YEAR (PF)" rows={report.by_year} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
