import { useEffect, useRef } from 'react'

// Background mein dheere chalte hue taare (sirf decoration)
export default function Starfield() {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    if (!ctx) return
    let stars: number[][] = []
    let raf = 0

    const size = () => {
      cv.width = window.innerWidth
      cv.height = window.innerHeight
      stars = Array.from({ length: 140 }, () => [
        Math.random() * cv.width, Math.random() * cv.height, Math.random() * 1.2 + 0.2, Math.random() * 6,
      ])
    }
    const loop = () => {
      ctx.clearRect(0, 0, cv.width, cv.height)
      for (const s of stars) {
        s[3] += 0.02
        ctx.fillStyle = `rgba(255,255,255,${0.3 + 0.3 * Math.sin(s[3])})`
        ctx.fillRect(s[0], s[1], s[2], s[2])
        s[1] += s[2] * 0.08
        if (s[1] > cv.height) s[1] = 0
      }
      raf = requestAnimationFrame(loop)
    }
    size()
    loop()
    window.addEventListener('resize', size)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', size)
    }
  }, [])

  return <canvas ref={ref} className="pointer-events-none fixed inset-0 z-0" />
}
