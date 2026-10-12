import { CandlestickSeries, ColorType, CrosshairMode, createChart, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type OrbRange } from '../lib/api'
import { ORB_COLORS, OrbPrimitive } from '../lib/orbPrimitive'
import { NY, addDays, fmtTick, fmtTs, nyToUtcString } from '../lib/time'
import { ORB_STATUS_COLOR } from './OrbPanel'

/*
  Research -> ORB: purane data (Dukascopy) pe ORB dabbe + backtest trades. Har din, har dabba: kyun trade hua / nahi hua.
  ML score walk-forward wala hai (model ne woh saal kabhi nahi dekha) -> 2019 se pehle ML nahi.
*/

const PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
const TFS = ['M5', 'M15', 'H1'] as const
const digitsOf = (s: string) => (s.endsWith('JPY') ? 3 : 5)
const STATUS_ORDER: Record<string, number> = { traded: 0, skip_daily: 1, skip_ml: 2, no_ml: 3, no_break: 4, skip_nocomp: 5 }

function lastMonday(): string {
  const d = new Date()
  d.setUTCDate(d.getUTCDate() - 7 - ((d.getUTCDay() + 6) % 7))
  return d.toISOString().slice(0, 10)
}

export default function OrbHistoryChart({ connected }: { connected: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const prim = useRef<OrbPrimitive | null>(null)
  const [symbol, setSymbol] = useState('GBPUSD')
  const [tf, setTf] = useState<(typeof TFS)[number]>('M15')
  const [date, setDate] = useState(lastMonday)
  const [days, setDays] = useState(5)
  const [ranges, setRanges] = useState<OrbRange[]>([])
  const [onlyBreaks, setOnlyBreaks] = useState(true)
  const [msg, setMsg] = useState<string | null>(null)
  const [loadedKey, setLoadedKey] = useState('')
  const key = `${symbol}|${tf}|${date}|${days}`

  useEffect(() => {
    if (!box.current) return
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#ffffff', fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", attributionLogo: false },
      grid: { vertLines: { color: 'rgba(196,208,224,0.10)' }, horzLines: { color: 'rgba(196,208,224,0.10)' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: 'rgba(154,174,198,0.16)' },
      timeScale: { borderColor: 'rgba(154,174,198,0.16)', timeVisible: true, tickMarkFormatter: (t: Time) => fmtTick(t as number, NY) },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY) },
    })
    const s = c.addSeries(CandlestickSeries, { upColor: '#00ff9c', downColor: '#ff426e', borderVisible: false, wickUpColor: '#00ff9c', wickDownColor: '#ff426e' })
    const p = new OrbPrimitive()
    s.attachPrimitive(p)
    chart.current = c
    series.current = s
    prim.current = p
    return () => { c.remove(); chart.current = null; series.current = null; prim.current = null }
  }, [])

  useEffect(() => {
    if (!connected) return
    let alive = true
    const start = nyToUtcString(addDays(date, -1), 17)        // Monday's Asia range starts Sunday 19:00 NY
    const end = nyToUtcString(addDays(date, days), 0)
    // naya pair / window: purana chart turant saaf + price axis auto-fit (warna purane pair ki price range pe atka rehta hai)
    series.current?.setData([])
    prim.current?.setRanges([], [])
    chart.current?.priceScale('right').applyOptions({ autoScale: true })
    Promise.all([api.candles({ symbol, tf, start, end, limit: 20000 }), api.orbResearchChart(symbol, start, end)])
      .then(([cs, orb]) => {
        if (!alive) return
        const dg = digitsOf(symbol)
        series.current?.applyOptions({ priceFormat: { type: 'price', precision: dg, minMove: 1 / 10 ** dg } })
        series.current?.setData(cs.map((c) => ({ time: c.time as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close })))
        prim.current?.setRanges(orb.ranges, cs.map((c) => c.time))
        chart.current?.priceScale('right').applyOptions({ autoScale: true })
        chart.current?.timeScale().fitContent()
        setRanges(orb.ranges)
        setMsg(cs.length ? null : 'No candles in this window (weekend / no data).')
        setLoadedKey(`${symbol}|${tf}|${date}|${days}`)
      })
      .catch((e: unknown) => { if (alive) { setMsg(e instanceof Error ? e.message : String(e)); setLoadedKey(`${symbol}|${tf}|${date}|${days}`) } })
    return () => { alive = false }
  }, [connected, symbol, tf, date, days])

  const rows = useMemo(() => ranges
    .filter((r) => !onlyBreaks || ['traded', 'skip_daily', 'skip_ml', 'no_ml'].includes(r.status))
    .sort((a, b) => (a.day === b.day ? (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9) : a.day < b.day ? -1 : 1)), [ranges, onlyBreaks])
  const traded = ranges.filter((r) => r.status === 'traded')
  const sumR = traded.reduce((s, r) => s + (r.r_net ?? 0), 0)

  const focus = (r: OrbRange) => {
    chart.current?.timeScale().setVisibleRange({ from: (r.t0 - 2 * 3600) as UTCTimestamp, to: ((r.exit_time ?? r.eod) + 2 * 3600) as UTCTimestamp })
  }

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>ORB HISTORY CHART</span>
        <em title="Downloaded Dukascopy data. ML scores are walk-forward (the model never saw that year), so before 2019 there is no ML score.">
          RANGES AND BACKTEST TRADES ON OLD DATA · WHY EACH RANGE WAS TRADED OR SKIPPED
        </em>
      </div>
      <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
        {PAIRS.map((p) => <button key={p} className={`chip ${p === symbol ? 'on' : ''}`} onClick={() => setSymbol(p)}>{p}</button>)}
        <span className="mx-1 text-muted">|</span>
        {TFS.map((t) => <button key={t} className={`chip ${t === tf ? 'on' : ''}`} onClick={() => setTf(t)}>{t}</button>)}
        <span className="mx-1 text-muted">|</span>
        <button className="chip" onClick={() => setDate(addDays(date, -7))} title="One week back">◀</button>
        <input type="date" value={date} onChange={(e) => e.target.value && setDate(e.target.value)}
          className="chip !py-0.5 text-[11px]" style={{ colorScheme: 'dark' }} title="First day (NY date)" />
        <button className="chip" onClick={() => setDate(addDays(date, 7))} title="One week forward">▶</button>
        {[3, 5, 10].map((n) => <button key={n} className={`chip ${days === n ? 'on' : ''}`} onClick={() => setDays(n)}>{n}D</button>)}
        <span className="ml-auto text-[11px]">
          <span className="text-muted">TRADES</span> {traded.length} <span className="text-muted">· NET</span>{' '}
          <b style={{ color: sumR >= 0 ? 'var(--green)' : 'var(--red)' }}>{sumR > 0 ? '+' : ''}{sumR.toFixed(2)}R</b>
        </span>
      </div>
      <div className="relative">
        <div ref={box} className="chart-surface h-[520px] w-full" />
        {(msg || loadedKey !== key) && <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px] text-muted">{loadedKey !== key ? `Loading ${symbol} ${tf}...` : msg}</div>}
      </div>
      <div className="mt-1 text-[9px] text-muted">
        <span style={{ color: ORB_COLORS.ASIA }}>ASIA</span> · <span style={{ color: ORB_COLORS.LON30 }}>LON30</span> · <span style={{ color: ORB_COLORS.LON60 }}>LON60</span> boxes.
        Faded = no compression. Dashed lines = breakout levels. Traded: <span style={{ color: '#00e5ff' }}>ENTRY</span> · <span style={{ color: '#ff426e' }}>SL</span> · <span style={{ color: '#00ff9c' }}>TP 2R</span> until the exit.
      </div>

      <div className="mt-3 flex items-center gap-2 text-[9px] tracking-[2px] text-muted">
        RANGES IN THIS WINDOW
        <button className={`chip !py-0 text-[9px] ${onlyBreaks ? 'on' : ''}`} onClick={() => setOnlyBreaks((v) => !v)}
          title="Only ranges that broke (traded / skipped by ML / 1-per-day)">ONLY BREAKS</button>
      </div>
      <div className="max-h-[300px] overflow-y-auto">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 text-[9px] text-muted" style={{ background: 'var(--panel)' }}><tr>
            <th className="text-left font-normal">DAY</th><th className="text-left font-normal">RANGE</th><th className="text-left font-normal">STATUS</th>
            <th className="text-left font-normal">WHY</th>
            <th className="text-right font-normal" title="Result at TP 2R in R (gross). For skipped breaks: what would have happened.">R (2R TP)</th>
            <th className="text-right font-normal">EXIT</th>
          </tr></thead>
          <tbody>{rows.map((r) => {
            const c = ORB_STATUS_COLOR[r.status] ?? 'var(--muted)'
            return (
              <tr key={`${r.day}${r.range}`} className="cursor-pointer border-t hover:bg-white/5" style={{ borderColor: 'var(--line)' }} onClick={() => focus(r)}
                title="Click to zoom the chart to this range">
                <td className="py-0.5">{r.day}</td>
                <td style={{ color: ORB_COLORS[r.range] }}>{r.range}</td>
                <td style={{ color: c }}>{r.status_text}</td>
                <td className="max-w-[420px] truncate text-muted" title={r.reason}>{r.reason}</td>
                <td className="text-right" style={{ color: r.r == null ? 'var(--muted)' : r.r > 0 ? 'var(--green)' : 'var(--red)', opacity: r.status === 'traded' ? 1 : 0.6 }}>
                  {r.r == null ? '—' : `${r.r > 0 ? '+' : ''}${r.r.toFixed(2)}`}{r.status !== 'traded' && r.r != null ? ' (what-if)' : ''}
                </td>
                <td className="text-right text-muted">{r.exit_reason ? r.exit_reason.toUpperCase() : '—'}</td>
              </tr>
            )
          })}</tbody>
        </table>
        {rows.length === 0 && <div className="py-2 text-[10px] text-muted">No breaks in this window.</div>}
      </div>
    </div>
  )
}
