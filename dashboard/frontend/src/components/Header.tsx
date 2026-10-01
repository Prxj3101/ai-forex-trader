import { useNow } from '../lib/hooks'
import Logo from './Logo'
import { THEMES, useTheme } from '../lib/theme'
import { IST, NY, clock, sessionAt } from '../lib/time'

type Props = { pair: string; connected: boolean; dataOk: boolean; mlOk: boolean }

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

// Theme picker: har gola ek color theme hai. Click karo, poora dashboard us color mein.
function ThemePicker() {
  const { theme, setTheme } = useTheme()
  return (
    <div className="flex items-center gap-1.5" title="Accent color">
      {THEMES.map((t) => (
        <button
          key={t.id}
          onClick={() => setTheme(t.id)}
          title={t.label}
          aria-label={`${t.label} theme`}
          className="h-3.5 w-3.5 cursor-pointer rounded-full border-0 p-0 transition-transform hover:scale-125"
          style={{
            background: t.accent,
            boxShadow: t.id === theme.id ? `0 0 0 2px #000, 0 0 0 3px ${t.accent}, 0 0 12px ${t.accent}` : `0 0 6px ${t.accent}`,
          }}
        />
      ))}
    </div>
  )
}

export default function Header({ pair, connected, dataOk, mlOk }: Props) {
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
        <Stat label="STRATEGY" value="ZONE TAP + ML" color="var(--cyan)" />
        <Stat label="NEW YORK" value={clock(NY, now)} color="var(--text)" />
        <Stat label="INDIA (IST)" value={clock(IST, now)} color="var(--text)" />
      </div>

      <div className="flex flex-wrap items-center justify-end gap-3">
        <Led label="ENGINE" color="var(--green)" on={connected} />
        <Led label="DATA" color="var(--green)" on={dataOk} speed="1.1s" />
        <Led label="BACKTEST" color="var(--amber)" on={connected} speed="2.2s" />
        <Led label="ML" color="var(--cyan)" on={mlOk} speed="1.8s" />
        <ThemePicker />
      </div>
    </header>
  )
}
