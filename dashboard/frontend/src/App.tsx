import { useCallback, useEffect, useMemo, useState } from 'react'
import AgentsPanel from './components/AgentsPanel'
import Boot from './components/Boot'
import ChartPanel, { type ChartSettings } from './components/ChartPanel'
import Header from './components/Header'
import MLBrain from './components/MLBrain'
import Pipeline from './components/Pipeline'
import ResearchPanel from './components/ResearchPanel'
import ScanPanel from './components/ScanPanel'
import SessionRadar from './components/SessionRadar'
import Starfield from './components/Starfield'
import Ticker from './components/Ticker'
import TradeLog from './components/TradeLog'
import { agentStates } from './lib/agents'
import { api, type AgentEvent, type AgentId, type ScanResult } from './lib/api'
import { useEvents, useNow } from './lib/hooks'

export default function App() {
  const [symbols, setSymbols] = useState<Record<string, string[]>>({})
  const [localEvents, setLocalEvents] = useState<AgentEvent[]>([])
  const [settings, setSettings] = useState<ChartSettings>({
    symbol: 'USDJPY', tf: 'M15', date: '2026-09-29', days: 2, swingN: 1, minDisp: 2,
  })
  const { events: wsEvents, connected } = useEvents()
  const now = useNow(500)

  // Kaunse pairs ka data hai (backend se)
  useEffect(() => {
    if (!connected) return
    api.symbols()
      .then((s) => {
        setSymbols(s)
        setSettings((cur) => (s[cur.symbol] ? cur : { ...cur, symbol: Object.keys(s)[0] ?? '' }))
      })
      .catch(() => setSymbols({}))
  }, [connected])

  // Dashboard ke apne events (jaise chart load) bhi log mein dikhao
  const onLog = useCallback((message: string, agent: AgentId = 'data', status: AgentEvent['status'] = 'done') => {
    setLocalEvents((prev) => [
      ...prev.slice(-99),
      { ts: Date.now() / 1000, agent, message, status, progress: status === 'active' ? 40 : 100, kind: 'log', data: {} },
    ])
  }, [])

  const events = useMemo(
    () => [...wsEvents, ...localEvents].sort((a, b) => a.ts - b.ts),
    [wsEvents, localEvents],
  )
  const states = agentStates(events, now)

  const lastResult = useMemo(() => {
    const r = [...wsEvents].reverse().find((e) => e.kind === 'result')
    return r ? (r.data as unknown as ScanResult) : null
  }, [wsEvents])

  // Scan chal raha hai? = aakhri "Loading ... history" event ke baad abhi tak result/error nahi aaya
  const scanRunning = useMemo(() => {
    let lastStart = -1
    let lastEnd = -1
    wsEvents.forEach((e, i) => {
      if (e.agent === 'data' && e.message.startsWith('Loading')) lastStart = i
      if (e.kind === 'result' || e.status === 'error') lastEnd = i
    })
    return lastStart > lastEnd && now / 1000 - wsEvents[lastStart].ts < 120
  }, [wsEvents, now])


  // Research: har naya "research ready" event aaye toh panel refresh ho
  const researchKey = useMemo(() => wsEvents.filter((e) => e.kind === 'research').length, [wsEvents])
  const researchBuilding = useMemo(() => {
    let start = -1
    let end = -1
    wsEvents.forEach((e, i) => {
      if (e.message.startsWith('Building research dataset')) start = i
      if (e.kind === 'research' || (e.status === 'error' && e.message.startsWith('Research'))) end = i
    })
    return start > end && now / 1000 - wsEvents[start].ts < 300
  }, [wsEvents, now])

  return (
    <>
      <Starfield />
      <div className="crt" />
      <Boot />

      <div className="relative z-[1] flex flex-col gap-2.5 p-3">
        <Header pair={settings.symbol || '---'} connected={connected} dataOk={Object.keys(symbols).length > 0} />
        <Ticker symbols={Object.keys(symbols)} />

        <div className="grid gap-2.5 xl:grid-cols-[250px_minmax(0,1fr)_300px]">
          <AgentsPanel states={states} />
          <ChartPanel symbols={symbols} settings={settings} onChange={setSettings} onLog={onLog} />
          <MLBrain />
        </div>

        <div className="grid gap-2.5 xl:grid-cols-[minmax(0,1.3fr)_280px_minmax(0,1fr)]">
          <TradeLog events={events} />
          <SessionRadar />
          <ScanPanel
            symbol={settings.symbol}
            tf={settings.tf}
            swingN={settings.swingN}
            minDisp={settings.minDisp}
            result={lastResult}
            running={scanRunning}
          />
        </div>

        <ResearchPanel
          symbols={symbols}
          defaultSymbol={settings.symbol || 'USDJPY'}
          refreshKey={researchKey}
          building={researchBuilding}
        />

        <Pipeline states={states} />

        {!connected && (
          <div className="text-center text-[11px] tracking-[2px]" style={{ color: 'var(--red)' }}>
            ENGINE OFFLINE // start the backend: uvicorn src.api.server:app --reload
          </div>
        )}
      </div>
    </>
  )
}
