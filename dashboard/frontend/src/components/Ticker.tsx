import { useEffect, useState } from 'react'
import { api } from '../lib/api'

type Quote = { symbol: string; price: number; change: number; digits: number }

// Upar scrolling strip: har pair ka latest close aur din ka % change (humare data se)
export default function Ticker({ symbols }: { symbols: string[] }) {
  const [quotes, setQuotes] = useState<Quote[]>([])

  useEffect(() => {
    if (!symbols.length) return
    Promise.all(
      symbols.map(async (s) => {
        const c = await api.candles({ symbol: s, tf: 'D1', limit: 2 })
        const last = c[c.length - 1]
        const prev = c[c.length - 2] ?? last
        const digits = last.close > 100 ? 3 : 5
        return { symbol: s, price: last.close, change: (100 * (last.close - prev.close)) / prev.close, digits }
      }),
    )
      .then(setQuotes)
      .catch(() => setQuotes([]))
  }, [symbols])

  const items = [...quotes, ...quotes, ...quotes]

  return (
    <div className="relative h-[26px] overflow-hidden" style={{ borderTop: '1px solid var(--line)', borderBottom: '1px solid var(--line)', background: 'rgba(6,6,8,0.9)' }}>
      <div className="ticker-track">
        {items.length === 0 && <span className="text-muted">NO MARKET DATA</span>}
        {items.map((q, i) => (
          <span key={i} className="mr-10 text-xs tracking-wider">
            <span style={{ color: 'var(--cyan)' }}>{q.symbol}</span>
            <b className="mx-2 font-normal">{q.price.toFixed(q.digits)}</b>
            <span style={{ color: q.change >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {q.change >= 0 ? '▲' : '▼'} {Math.abs(q.change).toFixed(2)}%
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}
