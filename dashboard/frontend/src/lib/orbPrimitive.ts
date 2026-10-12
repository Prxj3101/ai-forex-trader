import type {
  IChartApi,
  IPrimitivePaneRenderer,
  IPrimitivePaneView,
  ISeriesApi,
  ISeriesPrimitive,
  SeriesAttachedParameter,
  SeriesType,
  Time,
} from 'lightweight-charts'
import type { OrbRange } from './api'

/*
  ORB dabbe chart pe: Asia / London range ka box (high..low), uske baad breakout wali high / low lines
  (dashed, 11:00 NY tak), aur trade hua toh ENTRY / SL / TP lines 16:00 NY tak.
*/

type Target = Parameters<IPrimitivePaneRenderer['draw']>[0]
type Line = { x1: number; x2: number; y: number; color: string; dash: boolean; width: number; label?: string }
type Rect = { x1: number; x2: number; yTop: number; yBot: number; color: string; alpha: number; dashed: boolean; label: string; row: number }

export const ORB_COLORS: Record<string, string> = { ASIA: '#b388ff', LON30: '#ffb020', LON60: '#ff7a45' }
const ARMED_LIKE = new Set(['armed', 'no_break', 'skip_ml', 'skip_daily', 'skip_lot', 'traded'])

class OrbRenderer implements IPrimitivePaneRenderer {
  rects: Rect[] = []
  lines: Line[] = []

  draw(target: Target) {
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context
      const hr = scope.horizontalPixelRatio
      const vr = scope.verticalPixelRatio
      for (const r of this.rects) {
        const x = r.x1 * hr
        const w = Math.max((r.x2 - r.x1) * hr, 2)
        const y = r.yTop * vr
        const h = Math.max((r.yBot - r.yTop) * vr, 1)
        ctx.globalAlpha = r.alpha
        ctx.fillStyle = r.color + '1f'
        ctx.fillRect(x, y, w, h)
        ctx.strokeStyle = r.color
        ctx.lineWidth = 1.5 * hr
        if (r.dashed) ctx.setLineDash([5 * hr, 4 * hr])
        ctx.strokeRect(x, y, w, h)
        ctx.setLineDash([])
        if (r.label) {
          ctx.globalAlpha = Math.max(r.alpha, 0.8)
          ctx.font = `700 ${11 * vr}px 'IBM Plex Mono', monospace`
          ctx.strokeStyle = '#090b0e'
          ctx.lineWidth = 3 * hr
          ctx.fillStyle = r.color
          const ly = y + h + (13 + r.row * 13) * vr        // LON30 / LON60 same start -> labels on separate rows
          ctx.strokeText(r.label, x + 3 * hr, ly)
          ctx.fillText(r.label, x + 3 * hr, ly)
        }
        ctx.globalAlpha = 1
      }
      for (const l of this.lines) {
        ctx.globalAlpha = 0.95
        ctx.strokeStyle = l.color
        ctx.lineWidth = l.width * hr
        if (l.dash) ctx.setLineDash([6 * hr, 4 * hr])
        ctx.beginPath()
        ctx.moveTo(l.x1 * hr, l.y * vr)
        ctx.lineTo(Math.max(l.x2, l.x1 + 2) * hr, l.y * vr)
        ctx.stroke()
        ctx.setLineDash([])
        if (l.label) {
          ctx.font = `700 ${10 * vr}px 'IBM Plex Mono', monospace`
          ctx.strokeStyle = '#090b0e'
          ctx.lineWidth = 3 * hr
          ctx.fillStyle = l.color
          ctx.strokeText(l.label, l.x2 * hr - ctx.measureText(l.label).width - 3 * hr, l.y * vr - 3 * vr)
          ctx.fillText(l.label, l.x2 * hr - ctx.measureText(l.label).width - 3 * hr, l.y * vr - 3 * vr)
        }
        ctx.globalAlpha = 1
      }
    })
  }
}

class OrbView implements IPrimitivePaneView {
  rendererImpl = new OrbRenderer()
  renderer() {
    return this.rendererImpl
  }
  zOrder() {
    return 'bottom' as const
  }
}

export class OrbPrimitive implements ISeriesPrimitive<Time> {
  private chart: IChartApi | null = null
  private series: ISeriesApi<SeriesType> | null = null
  private requestUpdate: (() => void) | null = null
  private ranges: OrbRange[] = []
  private barTimes: number[] = []
  private visible = true
  private view = new OrbView()

  attached(param: SeriesAttachedParameter<Time>) {
    this.chart = param.chart
    this.series = param.series
    this.requestUpdate = param.requestUpdate
  }

  detached() {
    this.chart = null
    this.series = null
    this.requestUpdate = null
  }

  setRanges(ranges: OrbRange[], barTimes: number[]) {
    this.ranges = ranges
    this.barTimes = barTimes
    this.requestUpdate?.()
  }

  setVisible(v: boolean) {
    this.visible = v
    this.requestUpdate?.()
  }

  /** Time -> x. Chart ki candles ke andar: nearest candle; aakhri candle ke baad (future): logical index se. */
  private x(t: number): number | null {
    const bt = this.barTimes
    const ts = this.chart?.timeScale()
    if (!ts || !bt.length) return null
    if (t < bt[0]) return null
    const last = bt[bt.length - 1]
    if (t > last) {
      const step = bt.length > 1 ? bt[bt.length - 1] - bt[bt.length - 2] : 300
      return ts.logicalToCoordinate((bt.length - 1 + (t - last) / step) as never)
    }
    let lo = 0
    let hi = bt.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (bt[mid] < t) lo = mid + 1
      else hi = mid
    }
    return ts.timeToCoordinate(bt[lo] as Time)
  }

  updateAllViews() {
    const rd = this.view.rendererImpl
    rd.rects = []
    rd.lines = []
    const s = this.series
    if (!this.visible || !s || !this.chart || !this.barTimes.length) return
    const now = Date.now() / 1000
    for (const r of this.ranges) {
      if (r.hi == null || r.lo == null) continue
      const c = ORB_COLORS[r.range] ?? '#b388ff'
      const forming = r.status === 'forming'
      const x1 = this.x(r.t0)
      const x2 = this.x(forming ? Math.min(now, r.t1) : r.t1)
      const yTop = s.priceToCoordinate(r.hi)
      const yBot = s.priceToCoordinate(r.lo)
      if (x1 === null || x2 === null || yTop === null || yBot === null) continue
      const weak = r.status === 'skip_nocomp' || r.status === 'no_data'
      rd.rects.push({
        x1, x2, yTop, yBot, color: c, alpha: weak ? 0.35 : 1, dashed: forming,
        label: `${r.range} ${r.status_text}`,
        row: r.range === 'LON60' ? 1 : 0,
      })
      if (ARMED_LIKE.has(r.status)) {
        const end = r.break_time ?? Math.min(r.w_end, Math.max(now, r.t1))
        const xe = this.x(end)
        if (xe !== null) {
          rd.lines.push({ x1: x2, x2: xe, y: yTop, color: c, dash: true, width: 1 })
          rd.lines.push({ x1: x2, x2: xe, y: yBot, color: c, dash: true, width: 1 })
        }
      }
      if (r.status === 'traded' && r.break_time != null && r.entry != null && r.sl != null) {
        const risk = Math.abs(r.entry - r.sl)
        const tp = r.tp ?? (r.direction === 'bull' ? r.entry + 2 * risk : r.entry - 2 * risk)
        const xs = this.x(r.break_time)
        const xe = this.x(r.open ? Math.min(r.eod, Math.max(now, r.break_time)) : (r.exit_time ?? r.eod))
        const ye = s.priceToCoordinate(r.entry)
        const ys = s.priceToCoordinate(r.sl)
        const yt = s.priceToCoordinate(tp)
        if (xs !== null && xe !== null && ye !== null && ys !== null && yt !== null) {
          const side = r.direction === 'bull' ? 'BUY' : 'SELL'
          const res = r.r != null && !r.open ? ` ${r.r > 0 ? '+' : ''}${r.r.toFixed(2)}R` : ''
          rd.lines.push({ x1: xs, x2: xe, y: ye, color: '#00e5ff', dash: false, width: 2, label: `ORB ${side}${res}` })
          rd.lines.push({ x1: xs, x2: xe, y: ys, color: '#ff426e', dash: true, width: 1.5, label: 'SL' })
          rd.lines.push({ x1: xs, x2: xe, y: yt, color: '#00ff9c', dash: true, width: 1.5, label: 'TP 2R' })
        }
      }
    }
  }

  paneViews() {
    return [this.view]
  }
}
