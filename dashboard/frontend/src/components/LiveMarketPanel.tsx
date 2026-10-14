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
import { api, type LiveChart, type LiveStatus, type LiveTickTrade, type LiveTicks, type LiveWatchZone, type Zone } from '../lib/api'
import { floatPnl, fmtFill, fmtFillShort, fmtSigned, fmtUsd } from '../lib/pnl'
import ControlButton from './ControlButton'
import { useNow } from '../lib/hooks'
import { NY, fmtTick, fmtTs } from '../lib/time'
import { type Segment, SegmentsPrimitive } from '../lib/segmentsPrimitive'
import { ZonesPrimitive } from '../lib/zonesPrimitive'
import { ORB_COLORS, OrbPrimitive } from '../lib/orbPrimitive'

/*
  LIVE MARKET: bot ke apne MT5 (broker) candles, jo zones woh abhi dekh raha hai, aur is hafte ki news.
  Data VPS se aata hai (/api/live/chart + /api/live ka watch / health). Sirf dikhata hai.
*/

const CHART_POLL_MS = 5_000
const PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD']
const TFS = ['M5', 'M15', 'H1'] as const
const TF_SECONDS: Record<string, number> = { M5: 300, M15: 900, H1: 3600 }
const pipOf = (s: string) => (s.endsWith('JPY') ? 0.01 : 0.0001)
const digitsOf = (s: string) => (s.endsWith('JPY') ? 3 : 5)

const STATUS: Record<string, { text: string; color: string; tip: string }> = {
  ready: { text: 'READY', color: 'var(--green)', tip: 'Passes every check. If price touches the entry in this candle, the bot takes it.' },
  filled: { text: 'FILLED', color: 'var(--cyan)', tip: 'Price reached the entry: the trade is open (see OPEN TRADES)' },
  avoided: { text: 'AVOIDED', color: 'var(--amber)', tip: 'You told the bot not to trade this zone (press TAKE to trade it again)' },
  taken: { text: 'TAKEN (YOU)', color: 'var(--cyan)', tip: 'You pressed TAKE: the bot trades it even though ML / value area / quiet said no (risk rules still apply)' },
  fill_skipped: { text: 'FILL SKIPPED', color: 'var(--amber)', tip: 'Price reached the entry but a risk limit blocked the trade at that moment' },
  ml_low: { text: 'ML LOW', color: 'var(--muted)', tip: 'ML score below the top-30% threshold' },
  value_area: { text: 'VALUE AREA', color: 'var(--muted)', tip: 'Entry on the expensive side of the last 24h value area' },
  quiet: { text: 'QUIET', color: 'var(--muted)', tip: 'Market too quiet right now (tick volume below 0.9x the 24h average)' },
  news: { text: 'NEWS BLOCK', color: 'var(--amber)', tip: 'Inside the window around high-impact news' },
  limits: { text: 'LIMITS', color: 'var(--amber)', tip: 'Max open trades, already in this pair, or too much USD one way' },
  daily_limit: { text: 'DAILY RULE', color: 'var(--amber)', tip: 'Daily stop or daily risk budget reached' },
  kill_switch: { text: 'KILL SWITCH', color: 'var(--red)', tip: 'Kill switch is active' },
  lot_small: { text: 'LOT SMALL', color: 'var(--muted)', tip: 'Lot size below 0.01' },
  traded: { text: 'TRADED', color: 'var(--muted)', tip: 'This zone was already traded (one trade per zone)' },
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
    entry: z.entry,
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

function LiveChartBox({ symbol, tf, connected, ticks, showOrb = true }: { symbol: string; tf: string; connected: boolean; ticks: LiveTicks | null; showOrb?: boolean }) {
  const box = useRef<HTMLDivElement>(null)
  const liveBar = useRef<{ time: number; open: number; high: number; low: number; close: number } | null>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const zones = useRef<ZonesPrimitive | null>(null)
  const orbRef = useRef<OrbPrimitive | null>(null)
  const segsRef = useRef<SegmentsPrimitive | null>(null)
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
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#ffffff', fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", attributionLogo: false },
      grid: { vertLines: { color: 'rgba(196,208,224,0.10)' }, horzLines: { color: 'rgba(196,208,224,0.10)' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: 'rgba(154,174,198,0.16)' },
      timeScale: { borderColor: 'rgba(154,174,198,0.16)', timeVisible: true, tickMarkFormatter: (t: Time) => fmtTick(t as number, NY) },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY) },
    })
    const s = c.addSeries(CandlestickSeries, { upColor: '#00ff9c', downColor: '#ff426e', borderVisible: false, wickUpColor: '#00ff9c', wickDownColor: '#ff426e' })
    const p = new ZonesPrimitive()
    s.attachPrimitive(p)
    const sg = new SegmentsPrimitive()
    s.attachPrimitive(sg)
    const ob = new OrbPrimitive()
    s.attachPrimitive(ob)
    orbRef.current = ob
    segsRef.current = sg
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
    orbRef.current?.setRanges([], [])
    segsRef.current?.setSegments([], [])
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
    const barTimes = data.candles.map((c) => c.time)
    // zones: jo bot abhi dekh raha hai + jin zones se trade hua (trade ke baad bhi box rehta hai)
    const traded: Zone[] = (data.traded_zones ?? []).flatMap((z) => {
      if (z.top == null || z.bottom == null) return []
      return [{
        direction: z.direction, zone_candle_time: z.zone_candle_time ?? z.fill_time, created_at: z.created_at ?? z.fill_time,
        top: z.top, bottom: z.bottom, body_top: z.body_top ?? z.top, body_bottom: z.body_bottom ?? z.bottom,
        entry: z.entry, disp_atr: z.disp_atr ?? 0, has_fvg: z.has_fvg, first_touch_time: z.fill_time, first_touch_depth: null,
        invalidated_time: z.exit_ts, touches: 1,
      }]
    })
    zones.current?.setZones([...data.zones.map(toZone).filter((z): z is Zone => z !== null), ...traded], barTimes)
    orbRef.current?.setRanges(data.orb ?? [], barTimes)
    const lc = data.candles[data.candles.length - 1]
    liveBar.current = lc ? { ...lc } : null

    const first = data.candles.length ? data.candles[0].time : 0
    const mk: SeriesMarker<Time>[] = []
    for (const t of data.trades) {
      const buy = t.direction === 'bull'
      const tag = t.kind === 'orb' ? 'ORB ' : ''
      if (t.fill_ts >= first) mk.push({ time: t.fill_ts as UTCTimestamp, position: buy ? 'belowBar' : 'aboveBar', color: buy ? '#00ff9c' : '#ff426e', shape: buy ? 'arrowUp' : 'arrowDown', text: `${tag}${buy ? 'BUY' : 'SELL'}` })
      if (t.exit_ts >= first) mk.push({ time: t.exit_ts as UTCTimestamp, position: buy ? 'aboveBar' : 'belowBar', color: t.r > 0 ? '#00ff9c' : '#ff426e', shape: 'circle', text: `${t.r > 0 ? '+' : ''}${t.r.toFixed(1)}R` })
    }
    for (const t of data.open) {
      if (t.fill_time && t.fill_time >= first) {
        const buy = t.direction === 'bull'
        mk.push({ time: t.fill_time as UTCTimestamp, position: buy ? 'belowBar' : 'aboveBar', color: '#00e5ff', shape: buy ? 'arrowUp' : 'arrowDown', text: t.kind === 'orb' ? 'ORB OPEN' : 'OPEN' })
      }
    }
    mk.sort((a, b) => (a.time as number) - (b.time as number))
    markers.current?.setMarkers(mk)

    if (fresh && data.candles.length) {
      fitKey.current = key
      chart.current?.timeScale().setVisibleLogicalRange({ from: Math.max(0, data.candles.length - 120), to: data.candles.length + 3 })
    }
  }, [data, symbol, tf])

  useEffect(() => {
    orbRef.current?.setVisible(showOrb)
  }, [showOrb])

  // har ~1 s: chal rahi candle ko naye bhav se hilao (MT5 jaisa), 5 s wale chart poll ka wait nahi
  const tickBar = ticks?.forming[symbol]
  useEffect(() => {
    const s = series.current
    const lb = liveBar.current
    if (!s || !lb || !tickBar || !data || data.symbol !== symbol || data.tf !== tf) return
    const tfSec = TF_SECONDS[tf] ?? 300
    const [t0, o, h, l, c] = tickBar
    const bucket = Math.floor(t0 / tfSec) * tfSec
    if (bucket < lb.time) return                          // purana tick, chart aage hai
    const nb = bucket === lb.time
      ? (tf === 'M5' ? { time: bucket, open: o, high: h, low: l, close: c }
        : { time: bucket, open: lb.open, high: Math.max(lb.high, h), low: Math.min(lb.low, l), close: c })
      : { time: bucket, open: o, high: h, low: l, close: c }  // nayi candle shuru (agle chart poll pe pakka hogi)
    try {
      s.update({ ...nb, time: nb.time as UTCTimestamp })
      liveBar.current = nb
    } catch {
      // time ulta ho gaya (pair badla) -> agle chart poll pe theek
    }
  }, [tickBar, data, symbol, tf])

  // trade lines + live P&L label: data (har 5 s) ya naya bhav (har ~1 s) aane pe dobara
  const px = ticks?.prices[symbol]
  useEffect(() => {
    const s = series.current
    if (!s || !data || data.symbol !== symbol || data.tf !== tf) return
    const barTimes = data.candles.map((c) => c.time)
    // trade lines: ENTRY / SL / TP / TRAIL SL (sirf trade ke time tak, TradingView position tool jaisa)
    const GREEN = '#00ff9c'
    const RED = '#ff426e'
    const CYAN = '#00e5ff'
    const AMBER = '#ffb020'
    const exitTag = (exit: string | null | undefined) => (exit === 'trail' ? 'TRAIL' : exit === 'tp1' ? '1:1' : exit === 'tp2' ? '1:2' : '')
    const slLabel = (exit: string | null | undefined) => (exitTag(exit) ? `SL · ${exitTag(exit)}` : 'SL')
    const pnlOf = (t: (typeof data.open)[number]) => (px ? floatPnl({ ...t, symbol }, px.bid, px.ask) : null)
    // TP: bot ka diya hua (1:1) ya entry +/- 1R (trail trade / purana state jismein tp nahi)
    const tpOf = (t: (typeof data.open)[number]) =>
      t.tp ?? t.entry + (t.direction === 'bull' ? 1 : -1) * Math.abs(t.entry - t.sl)
    const segs: Segment[] = []
    for (const t of data.open) {
      if (!t.fill_time) continue
      const p = pnlOf(t)
      const side = `${t.kind === 'orb' ? `ORB ${t.range ?? ''} · ` : ''}${t.direction === 'bull' ? 'BUY' : 'SELL'} ${t.lots}`
      segs.push({
        t1: t.fill_time, t2: null, price: t.entry, color: CYAN, width: 2,
        label: p ? `${side} · ${fmtSigned(p.pips)} pips · ${fmtUsd(p.usd)} · ${fmtSigned(p.r, 2)}R` : `ENTRY ${side}`,
        labelColor: p ? (p.usd >= 0 ? GREEN : RED) : undefined,
      })
      segs.push({ t1: t.fill_time, t2: null, price: t.sl, color: RED, dash: true, label: slLabel(t.exit) })
      // TP line hamesha: 1:1 trade = asli target; TRAIL trade = +1R pe trailing shuru hota hai (fixed target nahi)
      const tpPx = tpOf(t)
      const atTp = floatPnl({ ...t, symbol }, tpPx, tpPx)
      segs.push({
        t1: t.fill_time, t2: null, price: tpPx, color: GREEN, dash: true,
        label: `${t.exit === 'trail' ? 'TP +1R · TRAIL STARTS' : t.exit === 'tp2' ? 'TP 1:2' : 'TP 1:1'} ${tpPx.toFixed(digitsOf(symbol))} (${fmtUsd(atTp.usd)})`,
      })
      if (t.stop_now != null && Math.abs(t.stop_now - t.sl) > 1e-9) {
        segs.push({ t1: t.fill_time, t2: null, price: t.stop_now, color: AMBER, width: 2, label: `TRAIL SL (best +${(t.best_r ?? 0).toFixed(1)}R)` })
      }
    }
    for (const t of data.trades) {
      const risk = Math.abs(t.entry - t.sl)
      const sign = t.direction === 'bull' ? 1 : -1
      segs.push({ t1: t.fill_ts, t2: t.exit_ts, price: t.entry, color: CYAN, faded: true })
      segs.push({ t1: t.fill_ts, t2: t.exit_ts, price: t.sl, color: RED, dash: true, faded: true })
      if (t.exit === 'tp1') segs.push({ t1: t.fill_ts, t2: t.exit_ts, price: t.entry + sign * risk, color: GREEN, dash: true, faded: true })
      else if (t.exit === 'tp2') segs.push({ t1: t.fill_ts, t2: t.exit_ts, price: t.tp ?? t.entry + 2 * sign * risk, color: GREEN, dash: true, faded: true })
    }
    for (const z of data.zones) {
      if (z.status !== 'ready' || z.created_at == null) continue        // READY zone: agar bhara toh yahan entry / SL
      const dg = digitsOf(symbol)
      segs.push({ t1: z.created_at, t2: null, price: z.entry, color: CYAN, dash: true, width: 1, label: `READY ${z.direction === 'bull' ? 'BUY' : 'SELL'} ${z.entry.toFixed(dg)}` })
      segs.push({ t1: z.created_at, t2: null, price: z.sl, color: RED, dash: true, width: 1, label: `${slLabel(z.exit)} ${z.sl.toFixed(dg)} (${z.sl_pips} pips)` })
    }
    segsRef.current?.setSegments(segs, barTimes)

    // price axis pe labels (line nahi, sirf right side ka tag)
    for (const l of lines.current) s.removePriceLine(l)
    lines.current = []
    for (const t of data.open) {
      const tag = (price: number, color: string, title: string) =>
        lines.current.push(s.createPriceLine({ price, color, lineVisible: false, lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: true, title }))
      const p = pnlOf(t)
      tag(t.entry, p ? (p.usd >= 0 ? GREEN : RED) : CYAN, p ? `ENTRY ${fmtUsd(p.usd)}` : 'ENTRY')
      tag(t.sl, RED, exitTag(t.exit) ? `SL ${exitTag(t.exit)}` : 'SL')
      tag(tpOf(t), GREEN, t.exit === 'trail' ? 'TP +1R' : t.exit === 'tp2' ? 'TP 1:2' : 'TP')
      if (t.stop_now != null && Math.abs(t.stop_now - t.sl) > 1e-9) tag(t.stop_now, AMBER, 'TRAIL')
    }
  }, [data, symbol, tf, px])

  const mine = data && data.symbol === symbol && data.tf === tf ? data : null

  // Candle countdown (TradingView jaisa): price axis pe last price ke neeche "kitna time bacha hai" is candle mein
  // text har second badalta hai; position har frame (taaki price scale drag / zoom pe label price ke saath chale)
  const [timer, setTimer] = useState<{ text: string; color: string } | null>(null)
  const timerEl = useRef<HTMLDivElement>(null)
  const last = mine?.candles[mine.candles.length - 1]
  const lastClose = last?.close
  useEffect(() => {
    const tfSec = TF_SECONDS[tf] ?? 300
    const tick = () => {
      if (!mine?.forming_time || !last) { setTimer(null); return }
      const left = last.time + tfSec - Math.floor(Date.now() / 1000)
      if (left < 0 || left > tfSec) { setTimer(null); return }
      const h = Math.floor(left / 3600)
      const m = Math.floor((left % 3600) / 60)
      const sec = left % 60
      const text = h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
      const lb = liveBar.current ?? last
      setTimer({ text, color: lb.close >= lb.open ? '#00ff9c' : '#ff426e' })
    }
    tick()
    const id = window.setInterval(tick, 1000)
    return () => window.clearInterval(id)
  }, [mine, last, tf])
  useEffect(() => {
    if (!timer || lastClose == null) return
    let raf = 0
    const place = () => {
      const el = timerEl.current
      const s = series.current
      const c = chart.current
      if (el && s && c) {
        const y = s.priceToCoordinate(liveBar.current?.close ?? lastClose)   // har second wala live bhav
        const hgt = box.current?.clientHeight ?? 0
        if (y === null || y < 0 || y > hgt - 40) el.style.display = 'none'
        else {
          el.style.display = 'block'
          el.style.top = `${y + 11}px`
          el.style.width = `${c.priceScale('right').width()}px`
        }
      }
      raf = requestAnimationFrame(place)
    }
    raf = requestAnimationFrame(place)
    return () => cancelAnimationFrame(raf)
  }, [timer, lastClose])

  return (
    <div className="relative">
      <div ref={box} className="chart-surface h-[580px] w-full" />
      {timer && (
        <div
          ref={timerEl}
          className="pointer-events-none absolute text-center text-[10px] leading-[14px]"
          style={{ right: 0, zIndex: 50, display: 'none', background: '#1b2026', color: '#ffffff', borderBottom: `2px solid ${timer.color}` }}
          title="Time left until this candle closes"
        >
          {timer.text}
        </div>
      )}
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

/** CLOSE button: khula shadow trade abhi market price pe band (bot ~1 s mein karta hai). */
export function CloseButton({ t, closing, pnl }: { t: LiveTickTrade; closing: boolean; pnl: string | null }) {
  if (t.fill_time == null) return null
  const orb = t.kind === 'orb'
  if (closing) return <span className="text-[9px]" style={{ color: 'var(--amber)' }}>CLOSING...</span>
  return (
    <ControlButton
      className="chip !px-1.5 !py-0 text-[9px]"
      style={{ color: 'var(--red)', borderColor: 'var(--red)' }}
      title="Close this shadow trade now at the market price"
      control={{ action: orb ? 'orb_close' : 'close', symbol: t.symbol, fill_time: t.fill_time! }}
      question={`Close this ${orb ? 'ORB ' : ''}trade now at the market price?\n${t.symbol} ${t.direction === 'bull' ? 'BUY' : 'SELL'} ${t.lots} lots at ${t.entry}${pnl ? ` · now ${pnl}` : ''}\nThis cannot be undone.`}
      confirmText="CLOSE NOW"
    >
      CLOSE
    </ControlButton>
  )
}

export default function LiveMarketPanel({ connected, data, ticks = null }: { connected: boolean; data: LiveStatus | null; ticks?: LiveTicks | null }) {
  const [symbol, setSymbol] = useState('EURUSD')
  const [tf, setTf] = useState<(typeof TFS)[number]>('M15')
  const [showOrb, setShowOrb] = useState(true)
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

  // khule trades: ticks (har ~1 s) se -> fill ke seconds baad dikhe; ticks na hon toh 15 s wala data
  const openTrades: LiveTickTrade[] = ticks?.updated ? ticks.open : (data?.open_trades ?? [])
  const floating = openTrades.reduce((s, t) => {
    const p = ticks?.prices[t.symbol]
    return s + (p ? floatPnl(t, p.bid, p.ask).usd : 0)
  }, 0)

  // manual buttons ki halat (ticks se, har ~1 s)
  const isAvoided = (sym: string, zone: number | null | undefined, dir: string) =>
    zone != null && (ticks?.avoid ?? []).some((a) => a.symbol === sym && a.zone === zone && a.direction === dir)
  const isClosing = (t: LiveTickTrade) =>
    (t.kind === 'orb' ? ticks?.orb_close_pending ?? [] : ticks?.close_pending ?? []).some((c) => c.symbol === t.symbol && c.fill_time === t.fill_time)
  const isTaken = (sym: string, zone: number | null | undefined, dir: string) =>
    zone != null && (ticks?.take ?? []).some((a) => a.symbol === sym && a.zone === zone && a.direction === dir)
  // exit shown in the watchlist: yours (ticks.exit_set, instant) else the exit brain's
  const exitOf = (z: LiveWatchZone & { symbol: string }) => {
    if (!ticks?.exit_set || z.zone_candle_time == null) return { exit: z.exit, mine: !!z.user_exit }
    const m = ticks.exit_set.find((a) => a.symbol === z.symbol && a.zone === z.zone_candle_time && a.direction === z.direction)
    return m ? { exit: m.exit, mine: true } : { exit: (z.exit_bot ?? z.exit) as 'tp1' | 'trail', mine: false }
  }
  const exitPending = (t: LiveTickTrade) =>
    (ticks?.exit_pending ?? []).find((c) => c.symbol === t.symbol && c.fill_time === t.fill_time)

  const news = h?.news
  const pairs = Object.keys(data?.watch?.symbols ?? {}).length ? Object.keys(data!.watch!.symbols) : PAIRS

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>LIVE MARKET</span>
        <em title="Candles come from the bot's own MT5 broker feed on the VPS, not from the historical data on this PC">BROKER FEED · ZONES THE BOT IS WATCHING · NEWS</em>
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_410px]">
        <div className="min-w-0">
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
            {pairs.map((s) => (
              <button key={s} className={`chip ${s === symbol ? 'on' : ''}`} onClick={() => setSymbol(s)}>{s}</button>
            ))}
            <span className="mx-1 text-muted">|</span>
            {TFS.map((t) => (
              <button key={t} className={`chip ${t === tf ? 'on' : ''}`} onClick={() => setTf(t)}>{t}</button>
            ))}
            <span className="mx-1 text-muted">|</span>
            <button className={`chip ${showOrb ? 'on' : ''}`} onClick={() => setShowOrb((v) => !v)}
              title="Show / hide the ORB ranges (Asia / London boxes) on the chart">ORB</button>
            {prices[symbol] && (
              <span className="ml-auto text-[11px]">
                <span className="text-muted">BID</span> {prices[symbol].bid.toFixed(digitsOf(symbol))} <span className="text-muted">ASK</span> {prices[symbol].ask.toFixed(digitsOf(symbol))}
              </span>
            )}
          </div>
          <LiveChartBox symbol={symbol} tf={tf} connected={connected} ticks={ticks} showOrb={showOrb} />
          <div className="mt-1 text-[9px] text-muted">
            Boxes = zones (bright = watching now, faded = already traded). Trade lines: <span style={{ color: '#00e5ff' }}>ENTRY</span> ·{' '}
            <span style={{ color: '#ff426e' }}>SL</span> · <span style={{ color: '#00ff9c' }}>TP (1:1 trades)</span> ·{' '}
            <span style={{ color: '#ffb020' }}>TRAIL SL (moves after +1R)</span>. Dotted = READY zone plan. Arrows = fills, circles = exits with R.
            {' '}ORB ranges: <span style={{ color: ORB_COLORS.ASIA }}>ASIA</span> · <span style={{ color: ORB_COLORS.LON30 }}>LON30</span> ·{' '}
            <span style={{ color: ORB_COLORS.LON60 }}>LON60</span> (dashed box = still forming, faded = no compression, dashed lines = breakout levels until 11:00 NY).
          </div>
        </div>

        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted" title="Every untouched zone the bot would trade if price reaches the entry. Sorted by how far price is from the entry.">
            WATCHLIST ({watch.length}) {data?.watch ? `· ML NEEDS ${Math.round(data.watch.threshold * 100)}%` : ''}
          </div>
          {watch.length === 0 ? (
            <div className="mb-3 text-[10px] text-muted">{data?.watch ? 'No live zones right now.' : 'Starts with the new bot version.'}</div>
          ) : (
            <div className="mb-3 max-h-[38vh] overflow-y-auto">
              <table className="watchlist-table w-full text-[11px]">
                <thead className="sticky top-0 text-[9px] text-muted" style={{ background: 'var(--panel)' }}>
                  <tr>
                    <th className="text-left font-normal">PAIR</th><th className="text-left font-normal">SIDE</th>
                    <th className="text-right font-normal">ENTRY</th>
                    <th className="text-right font-normal" title="Pips price still has to travel to reach the entry">AWAY</th>
                    <th className="text-right font-normal">ML</th>
                    <th className="text-right font-normal" title="Exit for this zone: 1:1 = fixed target at 1R; TRAIL = stop follows price after +1R. Click to switch (yours is marked *); click back to the brain's pick to undo.">EXIT</th>
                    <th className="text-right font-normal">STATUS</th>
                    <th className="font-normal" title="AVOID = do not trade this zone. TAKE = trade it (also when ML / value area / quiet said no). Press again to switch back - as often as you like, until price fills the zone." />
                  </tr>
                </thead>
                <tbody>
                  {watch.map((z, i) => {
                    const avoided = isAvoided(z.symbol, z.zone_candle_time, z.direction) || z.status === 'avoided'
                    const taken = !avoided && (ticks?.take ? isTaken(z.symbol, z.zone_candle_time, z.direction) : !!z.taken)
                    const takeable = ['ml_low', 'value_area', 'quiet'].includes(z.status)
                    const st = avoided ? STATUS.avoided : taken && (z.status === 'ready' || takeable) ? STATUS.taken
                      : STATUS[z.status] ?? { text: z.status.toUpperCase(), color: 'var(--muted)', tip: z.reason }
                    const ex = exitOf(z)
                    const side = z.direction === 'bull' ? 'BUY' : 'SELL'
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
                        <td className="text-right">
                          {z.zone_candle_time != null ? (
                            <ControlButton
                              className="cursor-pointer bg-transparent p-0 text-[11px] underline decoration-dotted"
                              style={{ color: ex.exit === 'trail' ? '#ffb020' : 'var(--green)' }}
                              title={`${ex.mine ? 'Your exit' : 'Exit brain'}: ${ex.exit === 'trail' ? 'TRAIL' : '1:1'}${z.exit_bot ? ` (brain picked ${z.exit_bot === 'trail' ? 'TRAIL' : '1:1'})` : ''}. Click to switch.`}
                              control={z.exit_bot != null && (ex.exit === 'trail' ? 'tp1' : 'trail') === z.exit_bot
                                ? { action: 'unexit', symbol: z.symbol, zone: z.zone_candle_time!, direction: z.direction }
                                : { action: 'exit', symbol: z.symbol, zone: z.zone_candle_time!, direction: z.direction, exit: ex.exit === 'trail' ? 'tp1' : 'trail' }}
                              question={`Switch the exit to ${ex.exit === 'trail' ? '1:1 (fixed target at 1R)' : 'TRAIL (stop follows price after +1R)'}?\n${z.symbol} ${side} at ${z.entry.toFixed(digitsOf(z.symbol))}${z.exit_bot ? ` · brain picked ${z.exit_bot === 'trail' ? 'TRAIL' : '1:1'}` : ''}\nYou can switch again any time.`}
                              confirmText={ex.exit === 'trail' ? 'SET 1:1' : 'SET TRAIL'}
                            >
                              {ex.exit === 'trail' ? 'TRAIL' : '1:1'}{ex.mine ? '*' : ''}
                            </ControlButton>
                          ) : '—'}
                        </td>
                        <td className="text-right" style={{ color: st.color }}>{st.text}</td>
                        <td className="pl-1 text-right">
                          {z.zone_candle_time != null && (taken || (takeable && !avoided)) && (
                            <ControlButton
                              className="chip !px-1.5 !py-0 text-[9px]"
                              style={{ color: taken ? 'var(--red)' : 'var(--cyan)', borderColor: taken ? 'var(--red)' : 'var(--cyan)' }}
                              title={taken ? 'Do NOT trade this zone (back to the bot decision: skip). You can switch again any time before the fill.' : 'Trade this zone even though the bot skips it (ML low / value area / quiet). Risk rules still apply. You can switch again any time before the fill.'}
                              control={{ action: taken ? 'untake' : 'take', symbol: z.symbol, zone: z.zone_candle_time!, direction: z.direction }}
                              question={taken
                                ? `Avoid this zone?\n${z.symbol} ${side} at ${z.entry.toFixed(digitsOf(z.symbol))}\nBack to the bot decision: it will NOT trade it. You can switch again.`
                                : `Take this zone?\n${z.symbol} ${side} at ${z.entry.toFixed(digitsOf(z.symbol))} · ML ${Math.round(z.score * 100)}%\nThe bot skips it (${STATUS[z.status]?.text ?? z.status}). With TAKE it trades it when price reaches the entry. You can switch again.`}
                              confirmText={taken ? 'AVOID' : 'TAKE'}
                            >
                              {taken ? 'AVOID' : 'TAKE'}
                            </ControlButton>
                          )}
                          {z.zone_candle_time != null && !taken && !(takeable && !avoided) && (
                            <ControlButton
                              className="chip !px-1.5 !py-0 text-[9px]"
                              style={{ color: avoided ? 'var(--cyan)' : 'var(--red)', borderColor: avoided ? 'var(--cyan)' : 'var(--red)' }}
                              title={avoided ? 'Trade this zone again (back to the bot decision: take). You can switch again any time before the fill.' : 'Tell the bot NOT to trade this zone. You can switch again any time before the fill.'}
                              control={{ action: avoided ? 'unavoid' : 'avoid', symbol: z.symbol, zone: z.zone_candle_time!, direction: z.direction }}
                              question={avoided
                                ? `Take this zone again?\n${z.symbol} ${side} at ${z.entry.toFixed(digitsOf(z.symbol))}\nBack to the bot decision: it trades it when price reaches the entry. You can switch again.`
                                : `Avoid this zone?\n${z.symbol} ${side} at ${z.entry.toFixed(digitsOf(z.symbol))}\nThe bot will not trade it. You can switch again.`}
                              confirmText={avoided ? 'TAKE' : 'AVOID'}
                            >
                              {avoided ? 'TAKE' : 'AVOID'}
                            </ControlButton>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="mb-1 flex items-center gap-2 text-[9px] tracking-[2px] text-muted" title="Open shadow trades. Shows up within seconds of the fill; profit / loss updates about every second.">
            OPEN TRADES ({openTrades.length})
            {openTrades.length > 0 && (
              <span className="text-[11px] tracking-[1px]" style={{ color: floating >= 0 ? 'var(--green)' : 'var(--red)' }}>· {fmtUsd(floating)}</span>
            )}
          </div>
          {openTrades.length === 0 ? (
            <div className="mb-3 text-[10px] text-muted">No open trades.</div>
          ) : (
            <div className="mb-3 flex flex-col gap-1">
              {openTrades.map((t, i) => {
                const p = ticks?.prices[t.symbol]
                const pl = p ? floatPnl(t, p.bid, p.ask) : null
                const col = pl ? (pl.usd >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--muted)'
                const dg = digitsOf(t.symbol)
                const commission = t.commission_usd ?? t.lots * (data?.account?.commission_per_lot ?? 5)
                const fresh = t.filled_at != null && now - t.filled_at < 300
                return (
                  <div
                    key={i}
                    className="cursor-pointer border-l-2 py-1 pl-2 text-[11px]"
                    style={{ borderColor: col, background: 'var(--panel2)' }}
                    onClick={() => setSymbol(t.symbol)}
                    title={`Filled ${fmtFill(t)} NY · SL ${t.sl.toFixed(dg)}${t.sl_pips != null ? ` (${t.sl_pips} pips)` : ''} · commission ${fmtUsd(-commission)}${t.score != null ? ` · ML ${Math.round(t.score * 100)}%` : ''}. Click to show on the chart.`}
                  >
                    <div className="flex items-baseline gap-2">
                      <b className="font-medium">{t.symbol}</b>
                      <span style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'} {t.lots}</span>
                      {t.kind === 'orb'
                        ? <span className="text-[10px]" style={{ color: '#b388ff' }} title="ORB shadow trade (separate $10k account): TP 2R, SL other side of the range, closes 16:00 NY">ORB {t.range} · 1:2</span>
                        : <span className="text-[10px]" style={{ color: t.exit === 'trail' ? '#ffb020' : 'var(--green)' }} title={t.user_exit ? `Your exit (brain picked ${t.exit_bot === 'trail' ? 'TRAIL' : '1:1'})` : 'Exit brain'}>{t.exit === 'trail' ? 'TRAIL' : '1:1'}{t.user_exit ? '*' : ''}</span>}
                      {t.kind !== 'orb' && t.fill_time != null && (exitPending(t)
                        ? <span className="text-[9px]" style={{ color: 'var(--amber)' }}>SWITCHING...</span>
                        : (
                          <ControlButton
                            className="chip !px-1.5 !py-0 text-[9px]"
                            style={{ color: t.exit === 'trail' ? 'var(--green)' : '#ffb020', borderColor: t.exit === 'trail' ? 'var(--green)' : '#ffb020' }}
                            title={t.exit === 'trail' ? 'Switch this trade to a fixed 1:1 target (if it is already above +1R it closes now)' : 'Switch this trade to TRAIL: no fixed target, stop follows price after +1R'}
                            control={{ action: 'trade_exit', symbol: t.symbol, fill_time: t.fill_time!, exit: t.exit === 'trail' ? 'tp1' : 'trail' }}
                            question={t.exit === 'trail'
                              ? ((t.r_now ?? 0) >= 1
                                ? `This trade is already above +1R (${(t.r_now ?? 0).toFixed(2)}R).\nSwitching to 1:1 closes it NOW at the market price.`
                                : `Switch this trade to 1:1?\n${t.symbol} · fixed target at +1R from now on. You can switch again.`)
                              : `Switch this trade to TRAIL?\n${t.symbol} · no fixed target, the stop follows price after +1R. You can switch again.`}
                            confirmText={t.exit === 'trail' ? ((t.r_now ?? 0) >= 1 ? 'CLOSE AT 1:1' : 'SET 1:1') : 'SET TRAIL'}
                          >
                            {t.exit === 'trail' ? '→ 1:1' : '→ TRAIL'}
                          </ControlButton>
                        ))}
                      {fresh && <span className="text-[9px]" style={{ color: 'var(--cyan)' }}>NEW</span>}
                      <span className="ml-auto font-medium" style={{ color: col }}>{pl ? fmtUsd(pl.usd) : '—'}</span>
                      <CloseButton t={t} closing={isClosing(t)} pnl={pl ? fmtUsd(pl.usd) : null} />
                    </div>
                    <div className="flex items-baseline gap-2 text-[10px] text-muted">
                      <span>{t.entry.toFixed(dg)} → {p ? (t.direction === 'bull' ? p.bid : p.ask).toFixed(dg) : '—'}</span>
                      <span>{fmtFillShort(t)} NY</span>
                      <span style={{ color: 'var(--red)' }} title="Round-trip commission for this open trade">COMM {fmtUsd(-commission)}</span>
                      <span className="ml-auto" style={{ color: col }}>{pl ? `${fmtSigned(pl.pips)} pips · ${fmtSigned(pl.r, 2)}R` : ''}</span>
                    </div>
                  </div>
                )
              })}
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
            <div className="flex max-h-[28vh] flex-col gap-1 overflow-y-auto pr-1">
              {news.events.map((e, i) => {
                const start = e.time - (news.block_before_min ?? 0) * 60
                const end = e.time + (news.block_after_min ?? 0) * 60
                const live = now >= start && now <= end
                const past = now > end
                return (
                  <div key={i} className="border-l-2 py-0.5 pl-2 text-[11px]" style={{ borderColor: live ? 'var(--amber)' : past ? 'var(--line)' : 'var(--cyan)', opacity: 1, background: 'var(--panel2)' }}>
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
