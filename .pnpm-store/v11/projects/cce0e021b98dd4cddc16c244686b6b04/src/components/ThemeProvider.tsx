import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { DEFAULT_THEME, THEMES, ThemeContext } from '../lib/theme'

const KEY = 'aft-theme'

function loadSaved(): string {
  try {
    return localStorage.getItem(KEY) ?? DEFAULT_THEME.id
  } catch {
    return DEFAULT_THEME.id
  }
}

// Poore app ko current theme deta hai, aur CSS variables (--accent-rgb) set karta hai
export default function ThemeProvider({ children }: { children: ReactNode }) {
  const [id, setId] = useState(loadSaved)
  const theme = THEMES.find((t) => t.id === id) ?? DEFAULT_THEME

  useEffect(() => {
    document.documentElement.style.setProperty('--accent-rgb', theme.rgb)
    try {
      localStorage.setItem(KEY, theme.id)   // agli baar bhi yahi theme khule
    } catch {
      /* private mode etc. mein storage na mile toh koi baat nahi */
    }
  }, [theme])

  const value = useMemo(() => ({ theme, setTheme: setId }), [theme])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}
