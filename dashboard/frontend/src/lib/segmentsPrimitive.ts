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

/*
  Chart pe chhoti horizontal lines (start time -> end time): trade ka ENTRY / SL / TP / TRAIL SL.
  Price lines poore chart pe failti hain; ye sirf trade ke time tak rehti hain (TradingView position tool jaisa).
*/

type Target = Parameters<IPrimitivePaneRenderer['draw']>[0]

export type Segment = {
  t1: number            // unix seconds
  t2: number | null     // null = chart ke aakhri candle tak
  price: number
  color: string
  dash?: boolean
  width?: number
  label?: string
  labelColor?: string   // label ka alag rang (jaise live P&L: hara / laal)
  faded?: boolean       // band ho chuke trade -> halka
}

type Drawn = { x1: number; x2: number; y: number; s: Segment }

class SegRenderer implements IPrimitivePaneRenderer {
  items: Drawn[] = []

  draw(target: Target) {
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context
      const hr = scope.horizontalPixelRatio
      const vr = scope.verticalPixelRatio
      for (const d of this.items) {
        const s = d.s
        ctx.globalAlpha = s.faded ? 0.65 : 1
        ctx.strokeStyle = s.color
        ctx.lineWidth = (s.width ?? 1.5) * hr
        ctx.setLineDash(s.dash ? [5 * hr, 4 * hr] : [])
        ctx.beginPath()
        ctx.moveTo(d.x1 * hr, d.y * vr)
        ctx.lineTo(Math.max(d.x2, d.x1 + 3) * hr, d.y * vr)
        ctx.stroke()
        ctx.setLineDash([])
        if (s.label) {
          ctx.globalAlpha = 1
          ctx.fillStyle = s.labelColor ?? s.color
          ctx.font = `700 ${12 * vr}px 'IBM Plex Mono', monospace`
          ctx.strokeStyle = '#090b0e'
          ctx.lineWidth = 3 * hr
          ctx.strokeText(s.label, (Math.max(d.x2, d.x1 + 3) + 4) * hr, (d.y + 3) * vr)
          ctx.fillText(s.label, (Math.max(d.x2, d.x1 + 3) + 4) * hr, (d.y + 3) * vr)
        }
        ctx.globalAlpha = 1
      }
    })
  }
}

class SegView implements IPrimitivePaneView {
  rendererImpl = new SegRenderer()
  renderer() {
    return this.rendererImpl
  }
  zOrder() {
    return 'top' as const
  }
}

export class SegmentsPrimitive implements ISeriesPrimitive<Time> {
  private chart: IChartApi | null = null
  private series: ISeriesApi<SeriesType> | null = null
  private requestUpdate: (() => void) | null = null
  private segs: Segment[] = []
  private barTimes: number[] = []
  private view = new SegView()

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

  setSegments(segs: Segment[], barTimes: number[]) {
    this.segs = segs
    this.barTimes = barTimes
    this.requestUpdate?.()
  }

  /** Time ko chart ki kisi candle pe snap karo (us candle ke andar ka time -> wahi candle). */
  private snap(t: number): number | null {
    const bt = this.barTimes
    if (!bt.length || t < bt[0] - 86400 * 7) return null
    if (t <= bt[0]) return bt[0]
    if (t >= bt[bt.length - 1]) return bt[bt.length - 1]
    let lo = 0
    let hi = bt.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (bt[mid] <= t) lo = mid
      else hi = mid - 1
    }
    return bt[lo]
  }

  updateAllViews() {
    const chart = this.chart
    const series = this.series
    const items: Drawn[] = []
    if (chart && series && this.barTimes.length) {
      const ts = chart.timeScale()
      const last = this.barTimes[this.barTimes.length - 1]
      for (const s of this.segs) {
        const a = this.snap(s.t1)
        const b = this.snap(s.t2 ?? last)
        if (a === null || b === null) continue
        const x1 = ts.timeToCoordinate(a as Time)
        const x2 = ts.timeToCoordinate(b as Time)
        const y = series.priceToCoordinate(s.price)
        if (x1 === null || x2 === null || y === null) continue
        items.push({ x1, x2, y, s })
      }
    }
    this.view.rendererImpl.items = items
  }

  paneViews() {
    return [this.view]
  }
}
