import { animate, motion } from 'framer-motion'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api, type ScanResult } from '../lib/api'

type Props = {
  symbol: string
  tf: string
  swingN: number
  minDisp: number
  result: ScanResult | null
  running: boolean
}

// Number jo 0 se value tak animate hota hai
function Counter({ value, decimals = 0, suffix = '' }: { value: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const controls = animate(0, value, {
      duration: 1.2,
      onUpdate: (v) => {
        if (ref.current) ref.current.textContent = (decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString()) + suffix
      },
    })
    return () => controls.stop()
  }, [value, decimals, suffix])
  return <span ref={ref}>0</span>
}

function Metric({ label, tip, color, children }: { label: string; tip: string; color: string; children: ReactNode }) {
  return (
    <div className="cursor-help border px-2 py-1.5" style={{ background: 'var(--panel2)', borderColor: 'var(--line)' }} title={tip}>
      <small className="block text-[9px] tracking-wider text-muted">{label}</small>
      <b className="font-display text-[17px] font-medium" style={{ color }}>{children}</b>
    </div>
  )
}

export default function ScanPanel({ symbol, tf, swingN, minDisp, result, running }: Props) {
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setError(null)
    try {
      await api.scan({ symbol, tf, swing_n: swingN, min_disp: minDisp })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="panel">
      <div className="ptitle"><span>ZONE SCAN</span><em>FULL HISTORY TEST</em></div>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <button className="btn" onClick={run} disabled={running}>
          {running ? 'SCANNING...' : 'RUN SCAN'}
        </button>
        <span className="text-[11px] text-muted">
          {symbol} · {tf} · swing {swingN} · disp {minDisp}x ATR
        </span>
      </div>
      {error && <div className="mb-2 text-[11px]" style={{ color: 'var(--red)' }}>{error}</div>}

      {result ? (
        <motion.div key={`${result.symbol}-${result.tf}-${result.zones}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="ZONES FOUND" tip="Orderflow zones detected across the full history" color="var(--purple)">
              <Counter value={result.zones} />
            </Metric>
            <Metric label="REVISITED" tip="Share of zones that price came back to at least once" color="var(--green)">
              <Counter value={result.touched_pct} decimals={1} suffix="%" />
            </Metric>
            <Metric label="INVALIDATED" tip="Share of zones where a candle later closed through the zone" color="var(--red)">
              <Counter value={result.invalidated_pct} decimals={1} suffix="%" />
            </Metric>
            <Metric label="WITH FVG" tip="Share of zones whose displacement left a fair value gap" color="var(--cyan)">
              <Counter value={result.with_fvg_pct} decimals={1} suffix="%" />
            </Metric>
          </div>
          <div className="mt-2 text-[10px] tracking-wider text-muted">
            {result.symbol} {result.tf} · {result.bull.toLocaleString()} bullish / {result.bear.toLocaleString()} bearish ·
            avg {result.avg_touches} touches per zone · computed in {result.seconds}s
          </div>
        </motion.div>
      ) : (
        <div className="text-[11px] text-muted">Run a scan to test the zone detector on the full history.</div>
      )}
    </div>
  )
}
