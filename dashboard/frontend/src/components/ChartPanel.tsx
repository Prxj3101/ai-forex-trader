import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { api, type Candle, type Zone } from '../lib/api'
import { NY, addDays, fmtTick, fmtTs, nyToUtcString } from '../lib/time'
import { rgba, useTheme } from '../lib/theme'
import { ZonesPrimitive } from '../lib/zonesPrimitive'

export type ChartSettings = {
  symbol: string
  tf: string
  date: string       // NY date, "YYYY-MM-DD"
  days: number
  swingN: number
  minDisp: number
}

type Props = {
  symbols: Record<string, string[]>
  settings: ChartSettings
  onChange: (s: ChartSettings) => void
  onLog: (message: string, agent?: 'data' | 'zone', status?: 'active' | 'done' | 'error') => void
}

const TFS = ['M5', 'M15', 'H1', 'H4', 'D1']

export default function ChartPanel({ symbols, settings, onChange, onLog }: Props) {
  const { theme } = useTheme()
  const boxRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const zonesRef = useRef<ZonesPrimitive | null>(null)
  const candlesRef = useRef<Candle[]>([])
  const [hud, setHud] = useState<Candle | null>(null)
  const [zones, setZones] = useState<Zone[]>([])
  // Kaunsi settings ka data load ho chuka hai. Settings badli aur data abhi nahi aaya = loading.
  const [loaded, setLoaded] = useState<{ key: string; error: string | null } | null>(null)
  const settingsKey = JSON.stringify(settings)
  const loading = !loaded || loaded.key !== settingsKey
  const error = loaded && loaded.key === settingsKey ? loaded.error : null

  // Chart ek hi baar banta hai
  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#c3cfdf',
        fontFamily: "'Share Tech Mono', monospace",
        attributionLogo: false,
      },
      grid: { vertLines: { color: 'rgba(0,240,255,0.10)' }, horzLines: { color: 'rgba(0,240,255,0.10)' } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: 'rgba(0,240,255,0.5)', labelBackgroundColor: '#003a40' },
        horzLine: { color: 'rgba(0,240,255,0.5)', labelBackgroundColor: '#003a40' },
      },
      rightPriceScale: { borderColor: 'rgba(0,240,255,0.4)' },
      timeScale: {
        borderColor: 'rgba(0,240,255,0.4)',
        timeVisible: true,
        tickMarkFormatter: (t: Time) => fmtTick(t as number, NY),
      },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY) },
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#00ff88', downColor: '#ff2e63', borderVisible: false,
      wickUpColor: '#00ff88', wickDownColor: '#ff2e63',
    })
    const prim = new ZonesPrimitive()
    series.attachPrimitive(prim)

    chart.subscribeCrosshairMove((p) => {
      if (!p.time) {
        setHud(candlesRef.current[candlesRef.current.length - 1] ?? null)
        return
      }
      const c = candlesRef.current.find((x) => x.time === p.time)
      if (c) setHud(c)
    })

    chartRef.current = chart
    seriesRef.current = series
    zonesRef.current = prim
    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      zonesRef.current = null
    }
  }, [])

  // Theme badle -> chart ke grid/crosshair ke colors update karo
  useEffect(() => {
    chartRef.current?.applyOptions({
      grid: { vertLines: { color: rgba(theme, 0.1) }, horzLines: { color: rgba(theme, 0.1) } },
      crosshair: {
        vertLine: { color: rgba(theme, 0.5), labelBackgroundColor: rgba(theme, 0.35) },
        horzLine: { color: rgba(theme, 0.5), labelBackgroundColor: rgba(theme, 0.35) },
      },
      rightPriceScale: { borderColor: rgba(theme, 0.4) },
      timeScale: { borderColor: rgba(theme, 0.4) },
    })
  }, [theme])

  // Settings badle -> naya data + zones load karo
  useEffect(() => {
    const { symbol, tf, date, days, swingN, minDisp } = settings
    if (!symbol) return
    let cancelled = false
    const key = JSON.stringify(settings)
    const start = nyToUtcString(date, 0)
    const end = nyToUtcString(addDays(date, days), 0)
    onLog(`Loading ${symbol} ${tf} candles for ${date} (+${days}d)`, 'data', 'active')

    Promise.all([
      api.candles({ symbol, tf, start, end, limit: 20000 }),
      api.zones({ symbol, tf, start, end, swing_n: swingN, min_disp: minDisp }),
    ])
      .then(([candles, zs]) => {
        if (cancelled) return
        candlesRef.current = candles
        seriesRef.current?.setData(
          candles.map((c) => ({ time: c.time as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close })),
        )
        zonesRef.current?.setZones(zs, candles.map((c) => c.time))
        chartRef.current?.timeScale().fitContent()
        setHud(candles[candles.length - 1] ?? null)
        setZones(zs)
        setLoaded({ key, error: null })
        onLog(`${candles.length.toLocaleString()} candles loaded`, 'data', 'done')
        const bull = zs.filter((z) => z.direction === 'bull').length
        onLog(`${zs.length} orderflow zones in view (${bull} demand, ${zs.length - bull} supply)`, 'zone', 'done')
      })
      .catch((e: unknown) => {
        if (cancelled) return
        const msg = e instanceof Error ? e.message : String(e)
        setLoaded({ key, error: msg })
        onLog(`Load failed: ${msg}`, 'data', 'error')
      })

    return () => {
      cancelled = true
    }
  }, [settings, onLog])

  const set = <K extends keyof ChartSettings>(k: K, v: ChartSettings[K]) => onChange({ ...settings, [k]: v })
  const tfs = symbols[settings.symbol] ?? TFS
  const digits = hud && hud.close > 100 ? 3 : 5

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>CHART // {settings.symbol || '---'}</span>
        <div className="flex flex-wrap items-center gap-2">
          <select className="field" value={settings.symbol} onChange={(e) => set('symbol', e.target.value)}>
            {Object.keys(symbols).map((s) => <option key={s}>{s}</option>)}
          </select>
          {TFS.filter((t) => tfs.includes(t)).map((t) => (
            <button key={t} className={`chip ${settings.tf === t ? 'on' : ''}`} onClick={() => set('tf', t)}>{t}</button>
          ))}
          <input type="date" className="field" value={settings.date} onChange={(e) => e.target.value && set('date', e.target.value)} title="Start date (New York)" />
          <select className="field" value={settings.days} onChange={(e) => set('days', Number(e.target.value))} title="How many days to show">
            {[1, 2, 3, 5, 10, 20].map((d) => <option key={d} value={d}>{d}D</option>)}
          </select>
        </div>
      </div>

      <div className="mb-2 flex flex-wrap items-center gap-4 text-[10px] text-muted">
        <label className="flex items-center gap-2" title="Swing size: 1 = internal structure (3 candles), 3 = major structure (7 candles)">
          SWING
          {[1, 2, 3].map((n) => (
            <button key={n} className={`chip ${settings.swingN === n ? 'on' : ''}`} onClick={() => set('swingN', n)}>{n}</button>
          ))}
        </label>
        <label className="flex items-center gap-2" title="Minimum displacement (in ATR) for a zone to count">
          MIN DISPLACEMENT
          <input type="range" min={1} max={4} step={0.25} value={settings.minDisp} onChange={(e) => set('minDisp', Number(e.target.value))} />
          <b className="w-10 font-normal" style={{ color: 'var(--cyan)' }}>{settings.minDisp.toFixed(2)}x</b>
        </label>
        <span>ZONES IN VIEW <b className="font-normal" style={{ color: 'var(--green)' }}>{zones.length}</b></span>
        <span>TIME: NEW YORK</span>
      </div>

      <div className="relative h-[430px]">
        <div ref={boxRef} className="absolute inset-0" />
        {hud && (
          <div className="pointer-events-none absolute left-2 top-1 z-10 text-[10px] leading-5" style={{ color: 'var(--cyan)', textShadow: `0 0 6px ${rgba(theme, 0.6)}` }}>
            {fmtTs(hud.time)} NY &nbsp; O {hud.open.toFixed(digits)} &nbsp;H {hud.high.toFixed(digits)} &nbsp;L {hud.low.toFixed(digits)} &nbsp;C {hud.close.toFixed(digits)}
            <br />SPREAD {hud.spread.toFixed(digits)}
          </div>
        )}
        <AnimatePresence>
          {(loading || error) && settings.symbol && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="font-display absolute inset-0 z-20 flex items-center justify-center text-sm tracking-[4px]"
              style={{ background: 'rgba(2,4,10,0.6)', color: error ? 'var(--red)' : 'var(--cyan)' }}
            >
              {error ? `ERROR: ${error}` : <span className="blink">LOADING MARKET DATA</span>}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="mt-2 flex flex-wrap gap-4 text-[10px] text-muted">
        <span><i className="mr-1.5 inline-block h-2.5 w-2.5 align-[-1px]" style={{ background: 'rgba(0,255,136,.35)', border: '1px solid var(--green)' }} />Demand zone</span>
        <span><i className="mr-1.5 inline-block h-2.5 w-2.5 align-[-1px]" style={{ background: 'rgba(255,46,99,.3)', border: '1px solid var(--red)' }} />Supply zone</span>
        <span>Dashed line = limit entry (zone body)</span>
        <span>Faded box = invalidated zone</span>
      </div>
    </div>
  )
}
