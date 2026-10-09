import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { api, type Candle, type MLScore, type Zone } from '../lib/api'
import { NY, addDays, fmtTick, fmtTs, nyToUtcString } from '../lib/time'
import { rgba, useTheme } from '../lib/theme'
import { ZonesPrimitive, type LabelMode } from '../lib/zonesPrimitive'

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
  onLog: (message: string, agent?: 'data' | 'zone' | 'ml', status?: 'active' | 'done' | 'error') => void
  mlKey: number
  botDisp?: number      // is pair ke liye bot ki displacement setting (src/config.py)
}

const TFS = ['M5', 'M15', 'H1', 'H4', 'D1']

// ML score -> marker color: green = bot lega (top 30% + value area OK + market active),
// grey = top 30% lekin entry mehengi (VA filter skip), blue-grey = top 30% lekin market shaant (activity filter skip),
// amber = top 50%, red = skip
const vaSkip = (s: MLScore) => s.keep_top30 && s.va_ok === false
const quietSkip = (s: MLScore) => s.keep_top30 && s.va_ok !== false && s.act_ok === false
const isTake = (s: MLScore) => s.keep_top30 && s.va_ok !== false && s.act_ok !== false
const scoreColor = (s: MLScore) =>
  vaSkip(s) ? '#8a94a6' : quietSkip(s) ? '#5c7c99' : s.keep_top30 ? '#00ff9c' : s.keep_top50 ? '#ffc400' : '#ff426e'

/** ML scores ko chart markers mein badlo. Fill time M5 ka hai -> us TF ki candle pe snap karo. */
function toMarkers(scores: MLScore[], times: number[]): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = []
  for (const s of scores) {
    let lo = 0
    let hi = times.length - 1
    let k = -1
    while (lo <= hi) {        // binary search: aakhri candle jiska time <= fill time
      const mid = (lo + hi) >> 1
      if (times[mid] <= s.fill_time) {
        k = mid
        lo = mid + 1
      } else hi = mid - 1
    }
    if (k < 0) continue
    const bull = s.direction === 'bull'
    out.push({
      time: times[k] as UTCTimestamp,
      position: bull ? 'belowBar' : 'aboveBar',
      shape: bull ? 'arrowUp' : 'arrowDown',
      color: scoreColor(s),
      text: `${Math.round(s.score * 100)}%${s.touch_number > 1 ? ` T${s.touch_number}` : ''}${
        vaSkip(s) ? ' VA SKIP' : quietSkip(s) ? ' QUIET' : isTake(s) && s.exit_ml ? (s.exit_ml === 'trail' ? ' TRAIL' : ' 1:1') : ''
      }`,
    })
  }
  return out.sort((a, b) => (a.time as number) - (b.time as number))
}

export default function ChartPanel({ symbols, settings, onChange, onLog, mlKey, botDisp }: Props) {
  const { theme } = useTheme()
  const boxRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const zonesRef = useRef<ZonesPrimitive | null>(null)
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)
  const candlesRef = useRef<Candle[]>([])
  const [hud, setHud] = useState<Candle | null>(null)
  const [zones, setZones] = useState<Zone[]>([])
  const [scores, setScores] = useState<MLScore[]>([])
  const [labelMode, setLabelMode] = useState<LabelMode>('live')
  const [showMl, setShowMl] = useState(true)
  const timesRef = useRef<number[]>([])
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
        textColor: '#ffffff', fontSize: 12,
        fontFamily: "'IBM Plex Mono', monospace",
        attributionLogo: false,
      },
      grid: { vertLines: { color: 'rgba(196,208,224,0.10)' }, horzLines: { color: 'rgba(196,208,224,0.10)' } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: 'rgba(0,240,255,0.5)', labelBackgroundColor: '#003a40' },
        horzLine: { color: 'rgba(0,240,255,0.5)', labelBackgroundColor: '#003a40' },
      },
      rightPriceScale: { borderColor: 'rgba(154,174,198,0.16)' },
      timeScale: {
        borderColor: 'rgba(154,174,198,0.16)',
        timeVisible: true,
        tickMarkFormatter: (t: Time) => fmtTick(t as number, NY),
      },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY) },
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#00ff9c', downColor: '#ff426e', borderVisible: false,
      wickUpColor: '#00ff9c', wickDownColor: '#ff426e',
    })
    const prim = new ZonesPrimitive()
    series.attachPrimitive(prim)
    const markers = createSeriesMarkers(series, [])

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
    markersRef.current = markers
    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      zonesRef.current = null
      markersRef.current = null
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
      api.mlScores({ symbol, tf, start, end }).catch(() => [] as MLScore[]),
    ])
      .then(([candles, zs, sc]) => {
        if (cancelled) return
        candlesRef.current = candles
        seriesRef.current?.setData(
          candles.map((c) => ({ time: c.time as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close })),
        )
        const times = candles.map((c) => c.time)
        timesRef.current = times
        zonesRef.current?.setZones(zs, times)
        chartRef.current?.timeScale().fitContent()
        setHud(candles[candles.length - 1] ?? null)
        setZones(zs)
        setScores(sc)
        setLoaded({ key, error: null })
        onLog(`${candles.length.toLocaleString()} candles loaded`, 'data', 'done')
        const bull = zs.filter((z) => z.direction === 'bull').length
        onLog(`${zs.length} orderflow zones in view (${bull} demand, ${zs.length - bull} supply)`, 'zone', 'done')
        if (sc.length) {
          const take = sc.filter(isTake).length
          onLog(`${sc.length} trades scored, ${take} in top 30% (would be taken)`, 'ml', 'done')
        }
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
  }, [settings, onLog, mlKey])

  // Labels / ML arrows toggle -> sirf drawing update (data dobara load nahi)
  useEffect(() => {
    zonesRef.current?.setLabelMode(labelMode)
  }, [labelMode])
  useEffect(() => {
    markersRef.current?.setMarkers(showMl ? toMarkers(scores, timesRef.current) : [])
  }, [scores, showMl])

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
        {botDisp !== undefined && (
          <button
            className={`chip ${settings.minDisp === botDisp && settings.swingN === 1 ? 'on' : ''}`}
            onClick={() => onChange({ ...settings, minDisp: botDisp, swingN: 1 })}
            title="The setting the bot and ML actually use for this pair. ML arrows only exist for these zones."
          >
            BOT SETTING {botDisp}x
          </button>
        )}
        <label className="flex items-center gap-2" title="Zone text labels: all zones, only live (not invalidated) zones, or none">
          LABELS
          {(['all', 'live', 'none'] as const).map((m) => (
            <button key={m} className={`chip ${labelMode === m ? 'on' : ''}`} onClick={() => setLabelMode(m)}>{m.toUpperCase()}</button>
          ))}
        </label>
        <button className={`chip ${showMl ? 'on' : ''}`} onClick={() => setShowMl((v) => !v)} title="Show or hide ML score arrows">
          ML ARROWS {showMl ? 'ON' : 'OFF'}
        </button>
        <span>ZONES IN VIEW <b className="font-normal" style={{ color: 'var(--green)' }}>{zones.length}</b></span>
        <span title="Trades in view with an out-of-sample ML score">
          ML SCORED <b className="font-normal" style={{ color: '#ffc400' }}>{scores.length}</b>
        </span>
        <span>TIME: NEW YORK</span>
      </div>

      <div className="chart-surface relative h-[430px]">
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
        <span><b style={{ color: "#00e5ff" }}>Cyan line</b> = limit entry (zone wick edge) · dotted = candle body</span>
        <span>Faded box = invalidated zone</span>
      </div>
      <div className="mt-1 flex flex-wrap gap-4 text-[10px] text-muted" title="Scores come from walk-forward testing: the model never saw that year during training. Available for M15/H1 from 2019, only for zones found with the BOT SETTING of this pair.">
        <span>ML arrows at fill:</span>
        <span style={{ color: '#00ff9c' }}>▲ top 30% (take)</span>
        <span style={{ color: '#8a94a6' }} title="ML liked it, but the entry was on the expensive side of the last 24h value area, so the bot skips it">▲ VA SKIP (entry too expensive)</span>
        <span style={{ color: '#5c7c99' }} title="ML liked it, but the market was quieter than usual right before the touch (last 15 min tick volume below 0.9x the 24h average), so the bot skips it">▲ QUIET (market inactive)</span>
        <span style={{ color: '#ffc400' }}>▲ top 50%</span>
        <span style={{ color: '#ff426e' }}>▲ low score (skip)</span>
        <span>T2/T3 = 2nd/3rd touch · 1:1 / TRAIL = exit chosen by ML · out-of-sample scores</span>
      </div>
    </div>
  )
}
