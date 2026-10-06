import { useCallback, useEffect, useMemo, useState } from 'react'
import AccountPanel from './components/AccountPanel'
import AgentsPanel from './components/AgentsPanel'
import Boot from './components/Boot'
import ChartPanel, { type ChartSettings } from './components/ChartPanel'
import Header from './components/Header'
import LivePanel from './components/LivePanel'
import MLBrain from './components/MLBrain'
import Pipeline from './components/Pipeline'
import ResearchPanel from './components/ResearchPanel'
import ScanPanel from './components/ScanPanel'
import SessionRadar from './components/SessionRadar'
import Starfield from './components/Starfield'
import Ticker from './components/Ticker'
import TradeLog from './components/TradeLog'
import { agentStates } from './lib/agents'
import { api, type AgentEvent, type AgentId, type BotConfig, type MLReport, type RiskReport, type ScanResult } from './lib/api'
import { useEvents, useNow } from './lib/hooks'

// Do tabs: LIVE (roz dekhna) aur RESEARCH (backtest / history / ML). Last chuna hua tab browser yaad rakhta hai.
type Tab = 'live' | 'research'
const TABS: { id: Tab; label: string; tip: string }[] = [
  { id: 'live', label: 'LIVE', tip: 'What the bot is doing right now on the VPS (broker feed)' },
  { id: 'research', label: 'RESEARCH', tip: 'History chart, ML brain, backtest account and research (downloaded data, not live)' },
]
const TAB_KEY = 'dashboard_tab'

function savedTab(): Tab {
  try {
    return localStorage.getItem(TAB_KEY) === 'research' ? 'research' : 'live'
  } catch {
    return 'live'
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>(savedTab)
  const changeTab = useCallback((t: Tab) => {
    setTab(t)
    try {
      localStorage.setItem(TAB_KEY, t)
    } catch {
      // storage band ho toh bas yaad nahi rahega
    }
  }, [])
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

  // Bot ki settings (src/config.py): har pair ka apna displacement. Chart slider usi pe start ho.
  const [botCfg, setBotCfg] = useState<BotConfig | null>(null)
  useEffect(() => {
    if (!connected) return
    api.config()
      .then((c) => {
        setBotCfg(c)
        setSettings((cur) => ({ ...cur, swingN: c.swing_n, minDisp: c.disp_by_symbol[cur.symbol] ?? c.default_disp }))
      })
      .catch(() => setBotCfg(null))
  }, [connected])

  // Pair badla -> slider us pair ki bot setting pe reset
  const onSettings = useCallback((s: ChartSettings) => {
    setSettings((cur) =>
      botCfg && s.symbol !== cur.symbol ? { ...s, minDisp: botCfg.disp_by_symbol[s.symbol] ?? botCfg.default_disp } : s,
    )
  }, [botCfg])
  const botDisp = botCfg ? botCfg.disp_by_symbol[settings.symbol] ?? botCfg.default_disp : undefined

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

  // ML Brain: har "trained" event pe report + chart scores refresh
  const mlKey = useMemo(() => wsEvents.filter((e) => e.kind === 'ml').length, [wsEvents])
  const [mlReport, setMlReport] = useState<MLReport | null>(null)
  useEffect(() => {
    if (!connected) return
    api.mlReport().then(setMlReport).catch(() => setMlReport(null))
  }, [connected, mlKey])

  // Risk Agent ka account simulation (training ke baad apne aap update hota hai)
  const [riskReport, setRiskReport] = useState<RiskReport | null>(null)
  useEffect(() => {
    if (!connected) return
    let cancelled = false
    let inFlight = false
    const refresh = async () => {
      if (inFlight) return
      inFlight = true
      try {
        const report = await api.riskReport()
        if (!cancelled) setRiskReport(report)
      } catch {
        // Keep the last report during a temporary connection failure.
      } finally {
        inFlight = false
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 30_000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [connected, mlKey])

  const mlTraining = useMemo(() => {
    let start = -1
    let end = -1
    wsEvents.forEach((e, i) => {
      if (e.agent === 'ml' && e.message.startsWith('Training ML Brain')) start = i
      if (e.kind === 'ml' || (e.agent === 'ml' && e.status === 'error')) end = i
    })
    return start > end && now / 1000 - wsEvents[start].ts < 600
  }, [wsEvents, now])

  const onTrain = useCallback(() => {
    const syms = mlReport?.symbols.length ? mlReport.symbols : ['EURUSD', 'USDJPY', 'GBPUSD']
    api.mlTrain(syms).catch((e: unknown) => onLog(`ML training: ${e instanceof Error ? e.message : String(e)}`, 'ml', 'error'))
  }, [mlReport, onLog])

  return (
    <>
      <Starfield />
      <Boot />

      <div className="dashboard-shell relative z-[1] flex flex-col">
        <Header
          pair={settings.symbol || '---'}
          connected={connected}
          dataOk={Object.keys(symbols).length > 0}
          mlOk={!!mlReport}
          tabs={
            <div className="header-tabs">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  className={`chip px-4 py-1.5 text-[12px] tracking-[3px] ${tab === t.id ? 'on' : ''}`}
                  onClick={() => changeTab(t.id)}
                  title={t.tip}
                  aria-pressed={tab === t.id}
                >
                  {t.label}
                </button>
              ))}
            </div>
          }
        />
        <Ticker symbols={Object.keys(symbols)} />

        {tab === 'live' ? (
          <>
            <LivePanel connected={connected} />
            <div className="grid gap-2.5 xl:grid-cols-[300px_minmax(0,1fr)]">
              <SessionRadar />
            </div>
          </>
        ) : (
          <>
            <div className="grid gap-2.5 xl:grid-cols-[250px_minmax(0,1fr)_300px]">
              <AgentsPanel states={states} />
              <ChartPanel symbols={symbols} settings={settings} onChange={onSettings} onLog={onLog} mlKey={mlKey} botDisp={botDisp} />
              <MLBrain report={mlReport} accountReport={riskReport} training={mlTraining} onTrain={onTrain} connected={connected} />
            </div>

            <AccountPanel report={riskReport} />

            <ResearchPanel
              symbols={symbols}
              defaultSymbol={settings.symbol || 'USDJPY'}
              refreshKey={researchKey}
              building={researchBuilding}
              coreSymbols={botCfg?.core_symbols ?? ['EURUSD', 'USDJPY', 'GBPUSD']}
            />

            <div className="grid gap-2.5 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
              <TradeLog events={events} />
              <ScanPanel
                symbol={settings.symbol}
                tf={settings.tf}
                swingN={settings.swingN}
                minDisp={settings.minDisp}
                result={lastResult}
                running={scanRunning}
              />
            </div>

            <Pipeline states={states} />
          </>
        )}

        {!connected && (
          <div className="text-center text-[11px] tracking-[2px]" style={{ color: 'var(--red)' }}>
            ENGINE OFFLINE // start the backend: uvicorn src.api.server:app --reload
          </div>
        )}
      </div>
    </>
  )
}
