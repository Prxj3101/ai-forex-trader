import { useId } from 'react'
import { useTheme } from '../lib/theme'

// An ascending A: the two strokes represent structure and a rising market.
export default function Logo({ size = 48 }: { size?: number }) {
  const id = useId().replace(/:/g, '')
  const accent = useTheme().theme.accent
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} className="logo-mark flex-none" role="img" aria-label="AI Forex Trader logo">
      <defs>
        <linearGradient id={`${id}-case`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#383d43" /><stop offset="1" stopColor="#1b1e22" />
        </linearGradient>
        <linearGradient id={`${id}-metal`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#f0f5f8" /><stop offset=".5" stopColor="#b8c2cc" /><stop offset="1" stopColor="#747f8c" />
        </linearGradient>
        <linearGradient id={`${id}-signal`} x1="0" y1="1" x2="1" y2="0">
          <stop stopColor="#358aa6" /><stop offset="1" stopColor={accent} />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="62" height="62" rx="16" fill={`url(#${id}-case)`} stroke="#555d67" strokeOpacity=".5" />
      <rect x="6" y="6" width="52" height="52" rx="12" fill="#1c2025" stroke="#111417" />
      <path d="M15 45 29 17h7l14 28h-9L32.5 27 24 45Z" fill={`url(#${id}-metal)`} />
      <path d="m24 41 7-7 5 3 11-12" fill="none" stroke="#181c21" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m24 41 7-7 5 3 11-12m-7 0h7v7" fill="none" stroke={`url(#${id}-signal)`} strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="18" cy="16" r="1.5" fill={accent} opacity=".8" />
    </svg>
  )
}
