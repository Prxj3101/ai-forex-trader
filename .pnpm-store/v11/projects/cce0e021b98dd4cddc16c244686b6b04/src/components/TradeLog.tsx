import { useEffect, useState } from 'react'
import type { AgentEvent } from '../lib/api'
import { agentColor, agentName } from '../lib/agents'
import { useTheme } from '../lib/theme'
import { NY, clock } from '../lib/time'

// Ek line jo typing effect ke saath likhi jati hai
function TypedLine({ event, animate }: { event: AgentEvent; animate: boolean }) {
  const { theme } = useTheme()
  const [n, setN] = useState(animate ? 0 : event.message.length)

  useEffect(() => {
    if (!animate) return
    const id = window.setInterval(() => {
      setN((v) => {
        if (v >= event.message.length) {
          window.clearInterval(id)
          return v
        }
        return v + 1
      })
    }, 14)
    return () => window.clearInterval(id)
  }, [animate, event.message])

  const color = event.status === 'error' ? 'var(--red)' : agentColor(theme, event.agent)
  return (
    <div className="truncate">
      <span className="text-muted">[{clock(NY, new Date(event.ts * 1000))}]</span>{' '}
      <span style={{ color }}>{agentName(event.agent)}</span> &gt;&gt; {event.message.slice(0, n)}
      {n < event.message.length && <span className="cursor" />}
    </div>
  )
}

export default function TradeLog({ events }: { events: AgentEvent[] }) {
  const shown = events.slice(-12)
  return (
    <div className="panel">
      <div className="ptitle"><span>TRADE LOG</span><em>LIVE AGENT ACTIVITY</em></div>
      <div className="flex h-[200px] flex-col justify-end overflow-hidden text-[11.5px] leading-[1.65]">
        {shown.length === 0 && <div className="text-muted">Waiting for activity...</div>}
        {shown.map((e, i) => (
          <TypedLine key={`${e.ts}-${i}-${e.message}`} event={e} animate={i === shown.length - 1} />
        ))}
      </div>
    </div>
  )
}
