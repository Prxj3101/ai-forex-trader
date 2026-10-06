import { Fragment } from 'react'
import { agentsFor, type AgentState } from '../lib/agents'
import { useTheme } from '../lib/theme'

// Decision flow: jo agent abhi kaam kar raha hai, uska step roshan hota hai
export default function Pipeline({ states }: { states: Record<string, AgentState> }) {
  const agents = agentsFor(useTheme().theme)
  return (
    <div className="panel">
      <div className="ptitle"><span>DECISION PIPELINE</span><em>RUNS ON EVERY CANDLE</em></div>
      <div className="flex flex-wrap items-center justify-between">
        {agents.map((a, i) => {
          const st = states[a.id]
          const cls = st.status === 'active' ? 'on' : st.status === 'offline' ? 'off' : ''
          return (
            <Fragment key={a.id}>
              <div className={`node ${cls}`} style={{ ['--c' as string]: a.color }}>
                <b className="font-display mb-0.5 block text-[11px] font-medium tracking-[2px]">
                  {a.name.replace(' AGENT', '')}
                </b>
                {a.pipeline}
              </div>
              {i < agents.length - 1 && <div className="link" />}
            </Fragment>
          )
        })}
      </div>
    </div>
  )
}
