// Time helpers. Backend sab kuch UTC seconds mein bhejta hai; dashboard NY aur IST mein dikhata hai.

export const NY = 'America/New_York'
export const IST = 'Asia/Kolkata'

const pad = (n: number) => String(n).padStart(2, '0')

export function clock(tz: string, d = new Date()): string {
  return d.toLocaleTimeString('en-GB', { timeZone: tz, hour12: false })
}

export function fmtTs(sec: number, tz = NY): string {
  return new Date(sec * 1000).toLocaleString('en-GB', {
    timeZone: tz, day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

export function fmtTick(sec: number, tz = NY): string {
  const d = new Date(sec * 1000)
  const hm = d.toLocaleTimeString('en-GB', { timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: false })
  if (hm === '00:00') return d.toLocaleDateString('en-GB', { timeZone: tz, day: '2-digit', month: 'short' })
  return hm
}

/** Kisi timezone ka UTC se offset (minutes mein) us moment pe. DST apne aap handle hota hai. */
function tzOffsetMin(date: Date, tz: string): number {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: tz, hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  }).formatToParts(date)
  const get = (t: string) => Number(parts.find((p) => p.type === t)?.value ?? 0)
  const asUtc = Date.UTC(get('year'), get('month') - 1, get('day'), get('hour'), get('minute'), get('second'))
  return (asUtc - date.getTime()) / 60000
}

/** "2026-09-29" + hour (NY local) -> backend ke liye UTC string "2026-09-29T04:00:00" */
export function nyToUtcString(dateStr: string, hour = 0): string {
  const naive = Date.parse(`${dateStr}T${pad(hour)}:00:00Z`)
  const offset = tzOffsetMin(new Date(naive), NY)
  return new Date(naive - offset * 60000).toISOString().slice(0, 19)
}

export function addDays(dateStr: string, days: number): string {
  const d = new Date(`${dateStr}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

export function hourIn(tz: string, d = new Date()): number {
  return Number(d.toLocaleString('en-GB', { timeZone: tz, hour: '2-digit', hour12: false }))
}

export type SessionName = 'Asian' | 'London' | 'London + NY' | 'New York' | 'Off-hours'

export function sessionAt(d = new Date()): SessionName {
  const tk = hourIn('Asia/Tokyo', d)
  const ld = hourIn('Europe/London', d)
  const ny = hourIn(NY, d)
  const inL = ld >= 8 && ld < 17
  const inN = ny >= 8 && ny < 17
  if (inL && inN) return 'London + NY'
  if (inL) return 'London'
  if (inN) return 'New York'
  if (tk >= 9 && tk < 18) return 'Asian'
  return 'Off-hours'
}

export const MARKETS = [
  { name: 'SYDNEY', tz: 'Australia/Sydney', open: 8, close: 17, lon: 151.2, lat: -33.9 },
  { name: 'TOKYO', tz: 'Asia/Tokyo', open: 9, close: 18, lon: 139.7, lat: 35.7 },
  { name: 'LONDON', tz: 'Europe/London', open: 8, close: 17, lon: -0.1, lat: 51.5 },
  { name: 'NEW YORK', tz: NY, open: 8, close: 17, lon: -74, lat: 40.7 },
] as const

export function isOpen(m: (typeof MARKETS)[number], d = new Date()): boolean {
  const h = hourIn(m.tz, d)
  const day = d.toLocaleDateString('en-US', { timeZone: m.tz, weekday: 'short' })
  if (day === 'Sat' || day === 'Sun') return false
  return h >= m.open && h < m.close
}
