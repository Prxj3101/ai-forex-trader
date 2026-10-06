import { AnimatePresence, motion } from 'framer-motion'
import { AGENTS, agentsFor, type AgentState } from '../lib/agents'
import { rgba, useTheme } from '../lib/theme'
import RobotHead from './RobotHead'

const LABEL: Record<AgentState['status'], string> = {
  active: 'ACTIVE', done: 'DONE', idle: 'IDLE', error: 'ERROR', offline: 'OFFLINE',
}

// Chhota equalizer: agent "soch" raha hai toh bars naachte hain, warna flat line
function ThinkingBars({ color, active }: { color: string; active: boolean }) {
  return (
    <div className="flex h-3 items-end gap-[2px]">
      {Array.from({ length: 10 }, (_, i) => (
        <i
          key={i}
          className="block w-[3px]"
          style={{
            background: color,
            height: active ? '100%' : '2px',
            opacity: active ? 0.9 : 0.35,
            boxShadow: active ? `0 0 6px ${color}` : 'none',
            animation: active ? `eq ${0.5 + (i % 4) * 0.15}s ease-in-out ${i * 0.06}s infinite alternate` : 'none',
            transformOrigin: 'bottom',
          }}
        />
      ))}
    </div>
  )
}

export default function AgentsPanel({ states }: { states: Record<string, AgentState> }) {
  const { theme } = useTheme()
  const agents = agentsFor(theme)
  const online = Object.values(states).filter((s) => s.status !== 'offline').length
  const anyActive = Object.values(states).some((s) => s.status === 'active')

  return (
    <div className="panel">
      <div className="ptitle">
        <span>AGENTS</span>
        <em className={anyActive ? 'blink' : ''} style={{ color: anyActive ? 'var(--green)' : undefined }}>
          {anyActive ? 'WORKING' : `${online} / ${AGENTS.length} ONLINE`}
        </em>
      </div>

      <div className="relative">
        {/* Data conduit: agents ko jodne wali line. Kaam chal raha ho toh isme data packets bahte hain */}
        <div className="absolute bottom-6 left-[22px] top-6 w-px" style={{ background: 'var(--line)' }}>
          {anyActive &&
            [0, 1, 2].map((i) => (
              <span key={i} className="conduit-packet" style={{ animationDelay: `${i * 0.6}s` }} />
            ))}
        </div>

        {agents.map((a) => {
          const st = states[a.id]
          const active = st.status === 'active'
          const lit = active || st.status === 'done'
          const color = st.status === 'error' ? '#ff2e63' : a.color

          return (
            <motion.div
              key={a.id}
              title={a.tip}
              className="relative mb-2 flex cursor-help items-center gap-2.5"
              animate={{ opacity: st.status === 'offline' ? 0.5 : 1 }}
            >
              <RobotHead color={color} status={st.status} variant={a.robot} />

              <motion.div
                className="relative min-w-0 flex-1 overflow-hidden border px-2.5 py-1.5"
                style={{ background: 'var(--panel2)' }}
                animate={{
                  borderColor: lit ? color : rgba(theme, 0.28),
                  boxShadow: lit ? `3px 3px 8px #101214, inset 0 1px 0 ${color}22` : '0 0 0 rgba(0,0,0,0)',
                }}
                transition={{ duration: 0.4 }}
              >
                {active && <span className="card-sweep" style={{ background: `linear-gradient(90deg, transparent, ${color}33, transparent)` }} />}
                <div className="flex items-center justify-between gap-2">
                  <span className="font-display text-[10.5px] tracking-[.5px]" style={{ color }}>{a.name}</span>
                  <span
                    className={`flex-none border px-1.5 text-[9px] tracking-wider ${active ? 'blink' : ''}`}
                    style={{ color: lit ? color : 'var(--muted)', borderColor: lit ? color : 'var(--line)' }}
                  >
                    {LABEL[st.status]}
                  </span>
                </div>

                <AnimatePresence mode="wait">
                  <motion.div
                    key={st.message}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 8 }}
                    transition={{ duration: 0.25 }}
                    className="my-1 min-h-[14px] truncate text-[11px]"
                    style={{ color: lit ? 'var(--text)' : 'var(--muted)' }}
                  >
                    {active && <span style={{ color }}>&gt; </span>}
                    {st.message}
                  </motion.div>
                </AnimatePresence>

                <div className="flex items-center gap-2">
                  <ThinkingBars color={color} active={active} />
                  <div className="bar flex-1" style={{ ['--c' as string]: color }}>
                    <i style={{ width: `${st.progress}%` }} />
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
