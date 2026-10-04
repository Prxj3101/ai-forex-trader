import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  createSeriesMarkers,
  LineStyle,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type LiveChart, type LiveStatus, type LiveWatchZone, type Zone } from '../lib/api'
import { useNow } from '../lib/hooks'
import { NY, fmtTick, fmtTs } from '../lib/time'
import { ZonesPrimitive } from '../lib/zonesPrimitive'

/*
  LIVE MARKET: bot ke apne MT5 (broker) candles, jo zones woh abhi dekh raha hai, aur is hafte ki news.
  Data VPS se aata hai (/api/live/chart + /api/live ka watch / health). Sirf dikhata hai.
*/

const CHART_POLL_MS = 5_000
const PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD']
const TFS = ['M5', 'M15', 'H1'] as const
const pipOf = (s: string) => (s.endsWith('JPY') ? 0.01 : 0.0001)
const digitsOf = (s: string) => (s.endsWith('JPY') ? 3 : 5)

const STATUS: Record<string, { text: string; color: string; tip: string }> = {
  ready: { text: 'READY', color: 'var(--green)', tip: 'Passes every check. If price touches the entry in this candle, the bot takes it.' },
  ml_low: { text: 'ML LOW', color: 'var(--muted)', tip: 'ML score below the top-30% threshold' },
  value_area: { text: 'VALUE AREA', color: 'var(--muted)', tip: 'Entry on the expensive side of the last 24h value area' },
  quiet: { text: 'QUIET', color: 'var(--muted)', tip: 'Market too quiet right now (tick volume below 0.9x the 24h average)' },
  news: { text: 'NEWS BLOCK', color: 'var(--amber)', tip: 'Inside the window around high-impact news' },
  limits: { text: 'LIMITS', color: 'var(--amber)', tip: 'Max open trades, already in this pair, or too much USD one way' },
  daily_limit: { text: 'DAILY RULE', color: 'var(--amber)', tip: 'Daily stop or daily risk budget reached' },
  kill_switch: { text: 'KILL SWITCH', color: 'var(--red)', tip: 'Kill switch is active' },
  lot_small: { text: 'LOT SMALL', color: 'var(--muted)', tip: 'Lot size below 0.01' },
}

function toZone(z: LiveWatchZone): Zone | null {
  if (z.top == null || z.bottom == null || z.created_at == null) return null
  return {
    direction: z.direction,
    zone_candle_time: z.zone_candle_time ?? z.created_at,
    created_at: z.created_at,
    top: z.top,
    bottom: z.bottom,
    body_top: z.body_top ?? z.top,
    body_bottom: z.body_bottom ?? z.bottom,
    disp_atr: z.disp_atr ?? 0,
    has_fvg: z.has_fvg,
    first_touch_time: null,
    first_touch_depth: null,
    invalidated_time: null,
    touches: 0,
  }
}

function untilText(sec: number) {
  const s = Math.abs(sec)
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.round((s % 3600) / 60)
  const t = d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`
  return sec >= 0 ? `IN ${t}` : `${t} AGO`
}

function LiveChartBox({ symbol, tf, connected }: { symbol: string; tf: string; connected: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const zones = useRef<ZonesPrimitive | null>(null)
  const markers = useRef<ISeriesMarkersPluginApi<Time> | null>(null)
  const lines = useRef<IPriceLine[]>([])
  const fitKey = useRef('')
  const [data, setData] = useState<LiveChart | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const el = box.current
    if (!el) return
    const c = createChart(el, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#c3cfdf', fontFamily: "'Share Tech Mono', monospace", attributionLogo: false },
      grid: { vertLines: { color: 'rgba(0,240,255,0.08)' }, horzLines: { color: 'rgba(0,240,255,0.08)' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: 'rgba(0,240,255,0.4)' },
      timeScale: { borderColor: 'rgba(0,240,255,0.4)', timeVisible: true, tickMarkFormatter: (t: Time) => fmtTick(t as number, NY) },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY) },
    })
    const s = c.addSeries(CandlestickSeries, { upColor: '#00ff88', downColor: '#ff2e63', borderVisible: false, wickUpColor: '#00ff88', wickDownColor: '#ff2e63' })
    const p = new ZonesPrimitive()
    s.attachPrimitive(p)
    chart.current = c
    series.current = s
    zones.current = p
    markers.current = createSeriesMarkers(s, [])
    return () => {
      c.remove()
      chart.current = null
      series.current = null
    }
  }, [])

  useEffect(() => {
    if (!connected) return
    let alive = true
    const load = () => {
      api.liveChart(symbol, tf)
        .then((d) => {
          if (!alive) return
          if (d.status === 'unreachable') setErr(d.error ?? 'VPS not reachable')
          else { setData(d); setErr(null) }
        })
        .catch((e: unknown) => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
    }
    // naya pair / TF: purana chart turant saaf karo (warna galat pair ka data dikhta rehta)
    series.current?.setData([])
    zones.current?.setZones([], [])
    markers.current?.setMarkers([])
    load()
    const id = window.setInterval(load, CHART_POLL_MS)
    return () => { alive = false; window.clearInterval(id) }
  }, [connected, symbol, tf])

  useEffect(() => {
    const s = series.current
    if (!s || !data || data.symbol !== symbol || data.tf !== tf) return
    const key = `${symbol}-${tf}`
    const fresh = fitKey.current !== key
    const d = digitsOf(symbol)
    s.applyOptions({ priceFormat: { type: 'price', precision: d, minMove: 1 / 10 ** d } })
    if (fresh) chart.current?.priceScale('right').applyOptions({ autoScale: true })   // naye pair ki price range pe fit
    s.setData(data.candles.map((c) => ({ ...c, time: c.time as UTCTimestamp })))
    zones.current?.setZones(data.zones.map(toZone).filter((z): z is Zone => z !== null), data.candles.map((c) => c.time))

    for (const l of lines.current) s.removePriceLine(l)
    lines.current = []
    for (const t of data.open) {
      const buy = t.direction === 'bull'
      lines.current.push(s.createPriceLine({ price: t.entry, color: buy ? '#00ff88' : '#ff2e63', lineWidth: 2, lineStyle: LineStyle.Solid, axisLabelVisible: true, title: `${buy ? 'BUY' : 'SELL'} ${t.lots}` }))
      lines.current.push(s.createPriceLine({ price: t.sl, color: '#ff2e63', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'SL' }))
    }

    const first = data.candles.length ? data.candles[0].time : 0
    const mk: SeriesMarker<Time>[] = []
    for (const t of data.trades) {
      const buy = t.direction === 'bull'
      if (t.fill_ts >= first) mk.push({ time: t.fill_ts as UTCTimestamp, position: buy ? 'belowBar' : 'aboveBar', color: buy ? '#00ff88' : '#ff2e63', shape: buy ? 'arrowUp' : 'arrowDown', text: buy ? 'BUY' : 'SELL' })
      if (t.exit_ts >= first) mk.push({ time: t.exit_ts as UTCTimestamp, position: buy ? 'aboveBar' : 'belowBar', color: t.r > 0 ? '#00ff88' : '#ff2e63', shape: 'circle', text: `${t.r > 0 ? '+' : ''}${t.r.toFixed(1)}R` })
    }
    for (const t of data.open) {
      if (t.fill_time && t.fill_time >= first) {
        const buy = t.direction === 'bull'
        mk.push({ time: t.fill_time as UTCTimestamp, position: buy ? 'belowBar' : 'aboveBar', color: '#00f0ff', shape: buy ? 'arrowUp' : 'arrowDown', text: 'OPEN' })
      }
    }
    mk.sort((a, b) => (a.time as number) - (b.time as number))
    markers.current?.setMarkers(mk)

    if (fresh && data.candles.length) {
      fitKey.current = key
      chart.current?.timeScale().setVisibleLogicalRange({ from: Math.max(0, data.candles.length - 120), to: data.candles.length + 3 })
    }
  }, [data, symbol, tf])

  const mine = data && data.symbol === symbol && data.tf === tf ? data : null
  return (
    <div className="relative">
      <div ref={box} className="h-[380px] w-full" />
      {mine?.forming_time && (
        <div className="pointer-events-none absolute left-2 top-1 flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--green)' }}
          title="The last candle is still forming and updates every few seconds">
          <i className="inline-block h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: 'var(--green)' }} /> LIVE
        </div>
      )}
      {(err || !mine || mine.candles.length === 0) && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px] text-muted">
          {err ?? (!mine ? `Loading ${symbol} ${tf}...` : `No live candles for ${symbol} yet. They appear after the bot checks this pair once.`)}
        </div>
      )}
    </div>
  )
}

export default function LiveMarketPanel({ connected, data }: { connected: boolean; data: LiveStatus | null }) {
  const [symbol, setSymbol] = useState('EURUSD')
  const [tf, setTf] = useState<(typeof TFS)[number]>('M15')
  const now = useNow(30_000) / 1000
  const h = data?.health
  const prices = useMemo(() => h?.prices ?? {}, [h?.prices])

  const watch = useMemo(() => {
    const rows: (LiveWatchZone & { symbol: string; away: number | null })[] = []
    for (const [sym, v] of Object.entries(data?.watch?.symbols ?? {})) {
      const p = prices[sym]
      for (const z of v.zones) {
        const away = p ? (z.direction === 'bull' ? (p.ask - z.entry) / pipOf(sym) : (z.entry - p.bid) / pipOf(sym)) : null
        rows.push({ ...z, symbol: sym, away })
      }
    }
    return rows.sort((a, b) => (a.away ?? 1e9) - (b.away ?? 1e9))
  }, [data?.watch, prices])

  const news = h?.news
  const pairs = Object.keys(data?.watch?.symbols ?? {}).length ? Object.keys(data!.watch!.symbols) : PAIRS

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>LIVE MARKET</span>
        <em title="Candles come from the bot's own MT5 broker feed on the VPS, not from the historical data on this PC">BROKER FEED · ZONES THE BOT IS WATCHING · NEWS</em>
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="min-w-0">
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
            {pairs.map((s) => (
              <button key={s} className={`chip ${s === symbol ? 'on' : ''}`} onClick={() => setSymbol(s)}>{s}</button>
            ))}
            <span className="mx-1 text-muted">|</span>
            {TFS.map((t) => (
              <button key={t} className={`chip ${t === tf ? 'on' : ''}`} onClick={() => setTf(t)}>{t}</button>
            ))}
            {prices[symbol] && (
              <span className="ml-auto text-[11px]">
                <span className="text-muted">BID</span> {prices[symbol].bid.toFixed(digitsOf(symbol))} <span className="text-muted">ASK</span> {prices[symbol].ask.toFixed(digitsOf(symbol))}
              </span>
            )}
          </div>
          <LiveChartBox symbol={symbol} tf={tf} connected={connected} />
          <div className="mt-1 text-[9px] text-muted">
            Boxes = zones the bot is watching right now (thick line = entry). Solid line = open shadow trade entry, dashed = its stop. Arrows = fills, circles = exits with R.
          </div>
        </div>

        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted" title="Every untouched zone the bot would trade if price reaches the entry. Sorted by how far price is from the entry.">
            WATCHLIST ({watch.length}) {data?.watch ? `· ML NEEDS ${Math.round(data.watch.threshold * 100)}%` : ''}
          </div>
          {watch.length === 0 ? (
            <div className="mb-3 text-[10px] text-muted">{data?.watch ? 'No live zones right now.' : 'Starts with the new bot version.'}</div>
          ) : (
            <div className="mb-3 max-h-[220px] overflow-y-auto">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 text-[9px] text-muted" style={{ background: 'var(--panel)' }}>
                  <tr>
                    <th className="text-left font-normal">PAIR</th><th className="text-left font-normal">SIDE</th>
                    <th className="text-right font-normal">ENTRY</th>
                    <th className="text-right font-normal" title="Pips price still has to travel to reach the entry">AWAY</th>
                    <th className="text-right font-normal">ML</th><th className="text-right font-normal">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {watch.map((z, i) => {
                    const st = STATUS[z.status] ?? { text: z.status.toUpperCase(), color: 'var(--muted)', tip: z.reason }
                    return (
                      <tr
                        key={i}
                        className="cursor-pointer"
                        onClick={() => setSymbol(z.symbol)}
                        title={`${z.reason || st.tip} · SL ${z.sl_pips} pips · exit ${z.exit === 'trail' ? 'trailing' : '1:1'} · ${z.lots} lots`}
                      >
                        <td>{z.symbol}</td>
                        <td style={{ color: z.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{z.direction === 'bull' ? 'BUY' : 'SELL'}</td>
                        <td className="text-right">{z.entry.toFixed(digitsOf(z.symbol))}</td>
                        <td className="text-right" style={{ color: z.away != null && z.away < 3 ? 'var(--amber)' : undefined }}>{z.away == null ? '—' : z.away.toFixed(1)}</td>
                        <td className="text-right" style={{ color: z.score >= (data?.watch?.threshold ?? 1) ? 'var(--green)' : 'var(--muted)' }}>{Math.round(z.score * 100)}%</td>
                        <td className="text-right" style={{ color: st.color }}>{st.text}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="mb-1 text-[9px] tracking-[2px] text-muted" title="High-impact news this week (ForexFactory). The bot opens no new trade on the affected pairs inside the block window.">
            HIGH-IMPACT NEWS THIS WEEK {news?.block_before_min != null ? `· BLOCK -${news.block_before_min} TO +${news.block_after_min} MIN` : ''}
          </div>
          {!news ? (
            <div className="text-[10px] text-muted">Starts with the new bot version.</div>
          ) : news.events.length === 0 ? (
            <div className="text-[10px] text-muted">No high-impact news this week.</div>
          ) : (
            <div className="flex max-h-[200px] flex-col gap-1 overflow-y-auto pr-1">
              {news.events.map((e, i) => {
                const start = e.time - (news.block_before_min ?? 0) * 60
                const end = e.time + (news.block_after_min ?? 0) * 60
                const live = now >= start && now <= end
                const past = now > end
                return (
                  <div key={i} className="border-l-2 py-0.5 pl-2 text-[11px]" style={{ borderColor: live ? 'var(--amber)' : past ? 'var(--line)' : 'var(--cyan)', opacity: past ? 0.45 : 1, background: 'var(--panel2)' }}>
                    <div className="flex items-baseline gap-2">
                      <span className="text-muted">{fmtTs(e.time)}</span>
                      <b className="font-medium">{e.currency}</b>
                      <span className="ml-auto text-[10px]" style={{ color: live ? 'var(--amber)' : 'var(--muted)' }}>{live ? 'BLOCKING NOW' : untilText(e.time - now)}</span>
                    </div>
                    <div className="text-[10px] text-muted">{e.event}{e.pairs.length ? ` · ${e.pairs.join(', ')}` : ''}</div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
