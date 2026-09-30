import { useEffect, useRef } from 'react'
import { rgba, useTheme } from '../lib/theme'

// ML model abhi train nahi hua. Panel honestly "NOT TRAINED" dikhata hai, animation dim rehti hai.
export default function MLBrain() {
  const { theme } = useTheme()
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const layers = [6, 8, 8, 4, 1]
    let pulses: { li: number; a: number; b: number; p: number }[] = []
    let raf = 0

    const draw = (t: number) => {
      const r = cv.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      cv.width = r.width * dpr
      cv.height = r.height * dpr
      const ctx = cv.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const W = r.width
      const H = r.height
      const pos = layers.map((n, li) =>
        Array.from({ length: n }, (_, i) => [12 + (li * (W - 24)) / (layers.length - 1), 10 + ((i + 0.5) * (H - 20)) / n]),
      )
      ctx.lineWidth = 0.6
      for (let li = 0; li < layers.length - 1; li++) {
        for (const a of pos[li]) {
          for (const b of pos[li + 1]) {
            ctx.strokeStyle = rgba(theme, 0.12)
            ctx.beginPath()
            ctx.moveTo(a[0], a[1])
            ctx.lineTo(b[0], b[1])
            ctx.stroke()
          }
        }
      }
      if (Math.random() < 0.15) {
        const li = Math.floor(Math.random() * (layers.length - 1))
        pulses.push({ li, a: Math.floor(Math.random() * layers[li]), b: Math.floor(Math.random() * layers[li + 1]), p: 0 })
      }
      pulses = pulses.filter((q) => q.p <= 1)
      for (const q of pulses) {
        const A = pos[q.li][q.a]
        const B = pos[q.li + 1][q.b]
        ctx.fillStyle = 'rgba(255,196,0,0.6)'
        ctx.beginPath()
        ctx.arc(A[0] + (B[0] - A[0]) * q.p, A[1] + (B[1] - A[1]) * q.p, 1.6, 0, Math.PI * 2)
        ctx.fill()
        q.p += 0.03
      }
      pos.forEach((l, li) =>
        l.forEach(([x, y]) => {
          const glow = 0.5 + 0.5 * Math.sin(t / 400 + x + y)
          ctx.fillStyle = li === layers.length - 1 ? '#9aa8bd' : rgba(theme, 0.5 + 0.4 * glow)
          ctx.beginPath()
          ctx.arc(x, y, li === layers.length - 1 ? 5 : 2.4, 0, Math.PI * 2)
          ctx.fill()
        }),
      )
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [theme])

  return (
    <div className="panel">
      <div className="ptitle"><span>ML BRAIN</span><em>ZONE SCORING</em></div>
      <div className="flex items-center gap-3">
        <div className="relative h-[110px] w-[110px] flex-none" title="Probability a zone reaches 1:2. Available after training.">
          <div className="absolute inset-0 rounded-full border border-dashed" style={{ borderColor: rgba(theme, 0.35), animation: 'spin 12s linear infinite' }} />
          <div className="absolute inset-[6px] rounded-full" style={{ border: '9px solid rgba(255,255,255,0.06)' }} />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <b className="font-display text-2xl text-muted">--</b>
            <small className="text-[9px] tracking-wider text-muted">P(WIN 1:2)</small>
          </div>
        </div>
        <div className="text-[11px] leading-6 text-muted">
          <div>STATUS <b className="font-normal" style={{ color: 'var(--amber)' }}>NOT TRAINED</b></div>
          <div>TRAINING DATA <b className="font-normal text-[var(--text)]">PENDING</b></div>
          <div>MODEL <b className="font-normal text-[var(--text)]">LightGBM</b></div>
        </div>
      </div>
      <canvas ref={ref} className="my-2 block h-[110px] w-full opacity-90" title="Neural activity: how the model combines features into a decision" />
      <div className="border px-3 py-2 text-center text-[11px] leading-5" style={{ borderColor: 'var(--line)', color: 'var(--muted)' }}>
        Zone scoring goes live after the zone dataset is labeled and the model is trained.
      </div>
    </div>
  )
}
