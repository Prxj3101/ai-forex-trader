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
import type { Zone } from './api'

/*
  TradingView Lightweight Charts mein rectangles (zones) ka built-in tool nahi hai.
  Isliye ek "primitive" (plugin) banaya hai jo chart ke upar zones ke glowing boxes draw karta hai.
  Chart zoom/scroll ho toh boxes bhi saath mein move hote hain.
*/

type Target = Parameters<IPrimitivePaneRenderer['draw']>[0]
type Box = { x1: number; x2: number; yTop: number; yBot: number; yBody: number; yEntry: number; color: string; label: string; dead: boolean }

/** Zone ke text labels: sab, sirf zinda (invalid nahi hue) zones ke, ya koi nahi. Chart saaf rakhne ke liye. */
export type LabelMode = 'all' | 'live' | 'none'

class ZonesRenderer implements IPrimitivePaneRenderer {
  boxes: Box[] = []

  draw(target: Target) {
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context
      const hr = scope.horizontalPixelRatio
      const vr = scope.verticalPixelRatio
      for (const b of this.boxes) {
        const x = b.x1 * hr
        const w = Math.max((b.x2 - b.x1) * hr, 2)
        const y = b.yTop * vr
        const h = Math.max((b.yBot - b.yTop) * vr, 1)
        ctx.globalAlpha = b.dead ? 0.55 : 1
        ctx.fillStyle = b.color + '26'
        ctx.fillRect(x, y, w, h)
        ctx.shadowColor = b.color
        ctx.shadowBlur = 4 * hr
        ctx.strokeStyle = b.color
        ctx.lineWidth = 1 * hr
        ctx.strokeRect(x, y, w, h)
        ctx.shadowBlur = 0
        // candle body edge: faint dotted line (reference only)
        ctx.globalAlpha = (b.dead ? 0.55 : 1) * 0.35
        ctx.setLineDash([2 * hr, 4 * hr])
        ctx.beginPath()
        ctx.moveTo(x, b.yBody * vr)
        ctx.lineTo(x + w, b.yBody * vr)
        ctx.stroke()
        ctx.setLineDash([])
        // ENTRY line = where the bot's limit order sits (zone wick edge)
        ctx.globalAlpha = b.dead ? 0.55 : 1
        ctx.strokeStyle = '#00e5ff'
        ctx.lineWidth = 2 * hr
        ctx.beginPath()
        ctx.moveTo(x, b.yEntry * vr)
        ctx.lineTo(x + w, b.yEntry * vr)
        ctx.stroke()
        ctx.strokeStyle = b.color
        ctx.lineWidth = 1 * hr
        if (b.label) {
          ctx.globalAlpha = 1
          ctx.fillStyle = b.color
          ctx.font = `700 ${12 * vr}px 'IBM Plex Mono', monospace`
          ctx.strokeStyle = '#090b0e'
          ctx.lineWidth = 3 * hr
          ctx.strokeText(b.label, x + 4 * hr, y - 4 * vr)
          ctx.fillText(b.label, x + 4 * hr, y - 4 * vr)
        }
        ctx.globalAlpha = 1
      }
    })
  }
}

class ZonesView implements IPrimitivePaneView {
  rendererImpl = new ZonesRenderer()
  renderer() {
    return this.rendererImpl
  }
  zOrder() {
    return 'bottom' as const
  }
}

export class ZonesPrimitive implements ISeriesPrimitive<Time> {
  private chart: IChartApi | null = null
  private series: ISeriesApi<SeriesType> | null = null
  private requestUpdate: (() => void) | null = null
  private zones: Zone[] = []
  private barTimes: number[] = []
  private labelMode: LabelMode = 'live'
  private view = new ZonesView()

  setLabelMode(mode: LabelMode) {
    this.labelMode = mode
    this.requestUpdate?.()
  }

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

  setZones(zones: Zone[], barTimes: number[]) {
    this.zones = zones
    this.barTimes = barTimes
    this.requestUpdate?.()
  }

  /** Time ko chart pe maujood kisi candle ke time pe snap karo (warna coordinate nahi milta). */
  private snap(t: number): number {
    const bt = this.barTimes
    if (!bt.length) return t
    if (t <= bt[0]) return bt[0]
    if (t >= bt[bt.length - 1]) return bt[bt.length - 1]
    let lo = 0
    let hi = bt.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (bt[mid] < t) lo = mid + 1
      else hi = mid
    }
    return bt[lo]
  }

  updateAllViews() {
    const chart = this.chart
    const series = this.series
    if (!chart || !series || !this.barTimes.length) {
      this.view.rendererImpl.boxes = []
      return
    }
    const ts = chart.timeScale()
    const last = this.barTimes[this.barTimes.length - 1]
    const boxes: Box[] = []
    for (const z of this.zones) {
      const start = this.snap(z.created_at)
      const end = this.snap(z.invalidated_time ?? last)
      const x1 = ts.timeToCoordinate(start as Time)
      const x2 = ts.timeToCoordinate(end as Time)
      const yTop = series.priceToCoordinate(z.top)
      const yBot = series.priceToCoordinate(z.bottom)
      const yBody = series.priceToCoordinate(z.direction === 'bull' ? z.body_top : z.body_bottom)
      const yEntry = series.priceToCoordinate(z.entry ?? (z.direction === 'bull' ? z.top : z.bottom))
      if (x1 === null || x2 === null || yTop === null || yBot === null || yBody === null || yEntry === null) continue
      const bull = z.direction === 'bull'
      const dead = z.invalidated_time !== null
      const showLabel = this.labelMode === 'all' || (this.labelMode === 'live' && !dead)
      boxes.push({
        x1, x2, yTop, yBot, yBody, yEntry,
        color: bull ? '#00ff9c' : '#ff426e',
        label: showLabel ? `${bull ? 'DEMAND' : 'SUPPLY'} ${z.bottom}-${z.top} · ${z.disp_atr}x ATR${z.has_fvg ? ' · FVG' : ''}` : '',
        dead,
      })
    }
    this.view.rendererImpl.boxes = boxes
  }

  paneViews() {
    return [this.view]
  }
}
