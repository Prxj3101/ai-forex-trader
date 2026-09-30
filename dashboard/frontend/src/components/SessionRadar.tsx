import { useEffect, useRef, useState } from 'react'
import { rgba, useTheme } from '../lib/theme'
import { MARKETS, isOpen } from '../lib/time'

// Ghoomta radar: duniya ke 4 bade forex markets, jo abhi khule hain wo green pulse karte hain
export default function SessionRadar() {
  const { theme } = useTheme()
  const ref = useRef<HTMLCanvasElement>(null)
  const [open, setOpen] = useState<Record<string, boolean>>({})

  useEffect(() => {
    const update = () => setOpen(Object.fromEntries(MARKETS.map((m) => [m.name, isOpen(m)])))
    update()
    const id = window.setInterval(update, 30000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
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
      const cx = W / 2
      const cy = H / 2
      const R = Math.min(W, H) / 2 - 8

      ctx.strokeStyle = rgba(theme, 0.32)
      ctx.lineWidth = 1
      for (const k of [1, 0.66, 0.33]) {
        ctx.beginPath()
        ctx.arc(cx, cy, R * k, 0, Math.PI * 2)
        ctx.stroke()
      }
      ctx.beginPath()
      ctx.moveTo(cx - R, cy)
      ctx.lineTo(cx + R, cy)
      ctx.moveTo(cx, cy - R)
      ctx.lineTo(cx, cy + R)
      ctx.stroke()

      const ang = (t / 1400) % (Math.PI * 2)
      for (let k = 0; k < 30; k++) {
        const a = ang - k * 0.025
        ctx.strokeStyle = `rgba(0,255,136,${0.5 * (1 - k / 30)})`
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a))
        ctx.stroke()
      }

      for (const m of MARKETS) {
        const a = ((m.lon + 180) / 360) * Math.PI * 2 - Math.PI / 2
        const rr = R * (0.35 + ((90 - Math.abs(m.lat)) / 90) * 0.55)
        const x = cx + rr * Math.cos(a)
        const y = cy + rr * Math.sin(a)
        const on = open[m.name]
        const col = on ? '#00ff88' : '#9aa8bd'
        ctx.fillStyle = col
        ctx.shadowColor = col
        ctx.shadowBlur = on ? 14 + 6 * Math.sin(t / 200) : 0
        ctx.beginPath()
        ctx.arc(x, y, on ? 4.5 : 3, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
        if (on) {
          ctx.strokeStyle = 'rgba(0,255,136,0.5)'
          ctx.beginPath()
          ctx.arc(x, y, 8 + 6 * ((t / 800) % 1), 0, Math.PI * 2)
          ctx.stroke()
        }
        ctx.font = "10px 'Share Tech Mono', monospace"
        ctx.fillText(m.name, x + 7, y - 6)
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [open, theme])

  return (
    <div className="panel">
      <div className="ptitle"><span>GLOBAL SESSIONS</span><em>OPEN MARKETS</em></div>
      <canvas ref={ref} className="block h-[210px] w-full" title="Global market sessions: which markets are open right now" />
      <div className="mt-1.5 flex justify-between text-[10px]">
        {MARKETS.map((m) => (
          <span key={m.name} style={{ color: open[m.name] ? 'var(--green)' : 'var(--muted)', textShadow: open[m.name] ? '0 0 6px var(--green)' : 'none' }}>
            ● {m.name}
          </span>
        ))}
      </div>
    </div>
  )
}
