import { createContext, useContext } from 'react'
import type { AgentId } from './api'

/*
  Dashboard ke color themes. "accent" = main color (pehle cyan tha).
  Green/red hamesha buy/sell ke liye rehte hain, wo theme se nahi badalte.
  Agar accent kisi agent ke color jaisa ho, toh us agent ko doosra color de dete hain (overrides).
*/
export type Theme = {
  id: string
  label: string
  accent: string   // hex, JS/SVG/canvas ke liye
  rgb: string      // "r, g, b", CSS rgba() ke liye
  overrides?: Partial<Record<AgentId, string>>
}

export const THEMES: Theme[] = [
  { id: 'cyan', label: 'Cyan', accent: '#00f0ff', rgb: '0, 240, 255' },
  { id: 'violet', label: 'Electric violet', accent: '#b14dff', rgb: '177, 77, 255', overrides: { structure: '#00f0ff' } },
  { id: 'gold', label: 'Gold', accent: '#ffb800', rgb: '255, 184, 0', overrides: { ml: '#ff7a1a' } },
  { id: 'blue', label: 'Electric blue', accent: '#3d8bff', rgb: '61, 139, 255' },
  { id: 'magenta', label: 'Magenta', accent: '#ff2bd6', rgb: '255, 43, 214', overrides: { risk: '#ff9f43' } },
]

export const DEFAULT_THEME = THEMES[0]

export type ThemeCtx = { theme: Theme; setTheme: (id: string) => void }

export const ThemeContext = createContext<ThemeCtx>({ theme: DEFAULT_THEME, setTheme: () => {} })

export const useTheme = () => useContext(ThemeContext)

/** accent ke saath transparency: rgba("r, g, b", 0.3) */
export const rgba = (theme: Theme, alpha: number) => `rgba(${theme.rgb}, ${alpha})`
