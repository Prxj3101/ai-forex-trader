import { useEffect, useState } from 'react'
import { WS_URL, type AgentEvent } from './api'

/** Har `ms` milliseconds pe current time deta hai (clocks, "abhi active hai?" check ke liye). */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), ms)
    return () => window.clearInterval(id)
  }, [ms])
  return now
}

/**
 * Backend ke WebSocket se live agent events sunta hai.
 * Connection toote toh har 2 second mein dobara jodne ki koshish karta hai.
 */
export function useEvents() {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let ws: WebSocket | null = null
    let stopped = false
    let retry: number | undefined

    const connect = () => {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!stopped) retry = window.setTimeout(connect, 2000)
      }
      ws.onmessage = (msg) => {
        const event = JSON.parse(msg.data) as AgentEvent
        setEvents((prev) => [...prev.slice(-299), event])
      }
    }
    connect()

    return () => {
      stopped = true
      window.clearTimeout(retry)
      ws?.close()
    }
  }, [])

  return { events, connected }
}
