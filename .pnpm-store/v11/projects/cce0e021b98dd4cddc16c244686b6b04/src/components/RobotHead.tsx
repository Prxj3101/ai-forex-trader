import type { AgentState } from '../lib/agents'

export type RobotVariant = 'dish' | 'twin' | 'antenna' | 'dome' | 'crest' | 'bolt'

/*
  Har agent ka apna robot head (original design, SVG se bana).
  ACTIVE  -> aankhein left-right scan karti hain, muh ke bars bolte hain, antenna blink, glow
  IDLE    -> aankhein kabhi kabhi blink karti hain
  OFFLINE -> aankhein band (grey lines), koi movement nahi
  Har agent ke sar pe alag "accessory" hai taaki pehchaan ho.
*/
export default function RobotHead({ color, status, variant }: { color: string; status: AgentState['status']; variant: RobotVariant }) {
  const active = status === 'active'
  const off = status === 'offline'
  const c = off ? '#4a5568' : color

  return (
    <svg
      viewBox="0 0 44 44"
      className="h-[44px] w-[44px] flex-none overflow-visible"
      style={{ filter: off ? 'none' : `drop-shadow(0 0 ${active ? 7 : 3}px ${c})` }}
    >
      {/* ---- accessory (sar ke upar) ---- */}
      {variant === 'dish' && (
        <g stroke={c} strokeWidth="1.5" fill="none">
          <path d="M14 8 Q22 14 30 8" />
          <line x1="22" y1="11" x2="22" y2="13" />
          <circle cx="22" cy="6" r="1.6" fill={c} className={active ? 'robot-blink-fast' : ''} />
        </g>
      )}
      {variant === 'twin' && (
        <g stroke={c} strokeWidth="1.5">
          <line x1="15" y1="12" x2="12" y2="5" />
          <line x1="29" y1="12" x2="32" y2="5" />
          <circle cx="12" cy="4.5" r="1.8" fill={c} className={active ? 'robot-blink-fast' : ''} />
          <circle cx="32" cy="4.5" r="1.8" fill={c} className={active ? 'robot-blink-fast delay' : ''} />
        </g>
      )}
      {variant === 'antenna' && (
        <g stroke={c} strokeWidth="1.5">
          <line x1="22" y1="12" x2="22" y2="4" />
          <circle cx="22" cy="3.5" r="2.2" fill={c} className={active ? 'robot-blink-fast' : ''} />
          <circle cx="22" cy="3.5" r="4.5" fill="none" strokeWidth="0.8" opacity="0.6" className={active ? 'robot-ping' : 'hidden'} />
        </g>
      )}
      {variant === 'dome' && (
        <g stroke={c} strokeWidth="1.2" fill="none">
          <path d="M11 13 Q22 0 33 13" />
          <circle cx="16" cy="9" r="1" fill={c} className={active ? 'robot-blink-fast' : ''} />
          <circle cx="22" cy="6.5" r="1" fill={c} className={active ? 'robot-blink-fast delay' : ''} />
          <circle cx="28" cy="9" r="1" fill={c} className={active ? 'robot-blink-fast' : ''} />
          <path d="M16 9 L22 6.5 L28 9" opacity="0.6" />
        </g>
      )}
      {variant === 'crest' && (
        <g stroke={c} strokeWidth="1.5" fill={c} fillOpacity="0.25">
          <path d="M16 12 L22 3 L28 12 Z" />
          <line x1="22" y1="6" x2="22" y2="11" />
        </g>
      )}
      {variant === 'bolt' && (
        <path d="M24 2 L18 10 L22 10 L19 15 L27 7 L23 7 Z" fill={c} className={active ? 'robot-blink-fast' : ''} />
      )}

      {/* ---- ears ---- */}
      <rect x="4" y="19" width="4" height="10" rx="1.5" fill={c} opacity="0.55" />
      <rect x="36" y="19" width="4" height="10" rx="1.5" fill={c} opacity="0.55" />

      {/* ---- head ---- */}
      <rect x="8" y="12" width="28" height="24" rx="6" fill="#07090d" stroke={c} strokeWidth="1.6" />
      {/* neck */}
      <rect x="17" y="36" width="10" height="4" rx="1" fill={c} opacity="0.5" />

      {/* ---- visor + eyes ---- */}
      <rect x="11" y="16.5" width="22" height="10" rx="4.5" fill="#000" stroke={c} strokeOpacity="0.45" strokeWidth="1" />
      {off ? (
        <g stroke={c} strokeWidth="1.5" strokeLinecap="round">
          <line x1="15" y1="21.5" x2="19" y2="21.5" />
          <line x1="25" y1="21.5" x2="29" y2="21.5" />
        </g>
      ) : (
        <g className={active ? 'robot-eyes-scan' : 'robot-eyes-blink'} style={{ transformOrigin: '22px 21.5px' }}>
          <circle cx="17" cy="21.5" r="2.3" fill={c} />
          <circle cx="27" cy="21.5" r="2.3" fill={c} />
          <circle cx="17.7" cy="20.8" r="0.7" fill="#fff" opacity="0.9" />
          <circle cx="27.7" cy="20.8" r="0.7" fill="#fff" opacity="0.9" />
        </g>
      )}

      {/* ---- mouth grill (bolte waqt naachta hai) ---- */}
      {[16, 20, 24, 28].map((x, i) => (
        <rect
          key={x}
          x={x - 1}
          y="29"
          width="2"
          height="4"
          rx="0.8"
          fill={c}
          opacity={off ? 0.4 : 0.85}
          style={{
            transformOrigin: `${x}px 33px`,
            animation: active ? `eq ${0.35 + i * 0.1}s ease-in-out ${i * 0.07}s infinite alternate` : 'none',
          }}
        />
      ))}
    </svg>
  )
}
