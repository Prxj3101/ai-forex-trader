/*
  AI FOREX TRADER ka logo (original design): trading + AI ka combo.
  - Bahar: hexagon (tech/AI ka shape) + ghoomta dashed ring
  - Andar: 3 candlesticks (trading)
  - Unke upar se ek "neural" trend line jiske joints glowing nodes hain (AI brain ke neurons),
    jo upar arrow pe khatam hoti hai (growth). Line baar baar khud draw hoti hai.
*/
import { useTheme } from '../lib/theme'

export default function Logo({ size = 48 }: { size?: number }) {
  const a = useTheme().theme.accent
  return (
    <svg viewBox="0 0 48 48" width={size} height={size} className="flex-none overflow-visible" style={{ filter: `drop-shadow(0 0 6px ${a})` }}>
      <g style={{ transformOrigin: '24px 24px', animation: 'spin 16s linear infinite' }}>
        <circle cx="24" cy="24" r="23.5" fill="none" stroke={a} strokeWidth="0.8" strokeDasharray="2 4" opacity="0.7" />
      </g>
      <path d="M24 2.5 L42.6 13.25 L42.6 34.75 L24 45.5 L5.4 34.75 L5.4 13.25 Z" fill="#030608" stroke={a} strokeWidth="1.6" />
      <path d="M24 6.5 L39.1 15.25 L39.1 32.75 L24 41.5 L8.9 32.75 L8.9 15.25 Z" fill="none" stroke={a} strokeWidth="0.5" opacity="0.35" />

      {/* candlesticks */}
      <g strokeWidth="1.2">
        <line x1="15" y1="25" x2="15" y2="37" stroke="#00ff88" />
        <rect x="13" y="28" width="4" height="7" fill="#00ff88" />
        <line x1="22" y1="21" x2="22" y2="33" stroke="#ff2e63" />
        <rect x="20" y="23" width="4" height="6" fill="#ff2e63" />
        <line x1="29" y1="14" x2="29" y2="29" stroke="#00ff88" />
        <rect x="27" y="16" width="4" height="10" fill="#00ff88" />
      </g>

      {/* neural trend line + nodes */}
      <polyline
        points="10,33 17,25 23,28 34,13"
        fill="none"
        stroke="#ffc400"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="logo-draw"
      />
      <path d="M34 13 L30 13.6 M34 13 L33.6 17" stroke="#ffc400" strokeWidth="1.6" strokeLinecap="round" />
      {[[10, 33], [17, 25], [23, 28], [34, 13]].map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="1.9" fill="#fff" stroke="#ffc400" strokeWidth="0.8"
          style={{ animation: `pulse 1.6s ease-in-out ${i * 0.3}s infinite` }} />
      ))}
    </svg>
  )
}
