import type { ReactNode } from 'react'
import { useNow } from '../lib/hooks'
import Logo from './Logo'
import { IST, NY, clock, sessionAt } from '../lib/time'

// tabs = LIVE / RESEARCH buttons (App se aate hain), top bar ke right side mein
type Props = { pair: string; connected: boolean; dataOk: boolean; mlOk: boolean; tabs?: ReactNode }

function Led({ label, color, on, speed = '1.4s' }: { label: string; color: string; on: boolean; speed?: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] tracking-wider text-muted">
      <i
        className="led"
        style={{ ['--c' as string]: on ? color : '#3a4660', animation: on ? `blink ${speed} infinite` : 'none' }}
      />
      {label}
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="text-center">
      <small className="block text-[9px] tracking-[2px] text-muted">{label}</small>
      <b className="font-display text-[17px] font-medium tracking-[2px]" style={{ color }}>{value}</b>
    </div>
  )
}

export default function Header({ pair, connected, dataOk, mlOk, tabs }: Props) {
  const now = new Date(useNow(1000))
  const session = sessionAt(now)

  return (
    <header className="hdr grid items-center gap-4 lg:grid-cols-[auto_1fr_auto]">
      <div className="flex items-center gap-3.5">
        <Logo size={46} />
        <div>
          <h1 className="font-display glow-cyan m-0 text-[17px] font-bold tracking-[4px]">AI FOREX TRADER</h1>
          <small className="mt-0.5 block text-[10px] tracking-[3px] text-muted">TRADING COMMAND CENTER // v0.1</small>
        </div>
      </div>

      <div className="flex flex-wrap justify-center gap-7">
        <Stat label="ACTIVE PAIR" value={pair} color="var(--amber)" />
        <Stat label="MARKET SESSION" value={session.toUpperCase()} color={session === 'Off-hours' ? 'var(--muted)' : 'var(--green)'} />
        <Stat label="NEW YORK" value={clock(NY, now)} color="var(--text)" />
        <Stat label="INDIA (IST)" value={clock(IST, now)} color="var(--text)" />
      </div>

      <div className="flex flex-wrap items-center justify-end gap-3">
        <Led label="ENGINE" color="var(--green)" on={connected} />
        <Led label="DATA" color="var(--green)" on={dataOk} speed="1.1s" />
        <Led label="BACKTEST" color="var(--amber)" on={connected} speed="2.2s" />
        <Led label="ML" color="var(--cyan)" on={mlOk} speed="1.8s" />
        {tabs}
      </div>
    </header>
  )
}
