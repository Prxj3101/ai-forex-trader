import { ColorType, LineSeries, createChart, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from 'lightweight-charts'
import { useEffect, useRef, useState } from 'react'
import { api, type OrbResearchFilters, type OrbResearchReport, type OrbStat } from '../lib/api'
import { ORB_COLORS } from '../lib/orbPrimitive'
import { NY, fmtTs } from '../lib/time'

/*
  Research -> ORB filters: khud combination chuno (pair, session, compression, ML, TP, 1 per day, direction, saal)
  -> turant PF / trades / equity. Sab net of costs ($5/lot + 0.2 pip on stops). Laptop ka downloaded data.
*/

const PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
const RANGES = ['ASIA', 'LON30', 'LON60']
const COMP: { id: string; label: string; tip: string }[] = [
  { id: 'nr4_or_tight', label: 'NR4 OR TIGHT', tip: 'Live rule: yesterday NR4 or range in the smallest 30% (vs last 20)' },
  { id: 'nr4', label: 'NR4', tip: "Yesterday's daily range was the smallest of the last 4 days" },
  { id: 'nr7', label: 'NR7', tip: "Yesterday's daily range was the smallest of the last 7 days" },
  { id: 'tight', label: 'TIGHT', tip: 'Range in the smallest 30% vs the last 20 same ranges' },
  { id: 'inside', label: 'INSIDE DAY', tip: 'Yesterday was inside the day before' },
  { id: 'any', label: 'NO FILTER', tip: 'Every breakout, compressed or not' },
  { id: 'none', label: 'ONLY NOT COMPRESSED', tip: 'The opposite: only ranges WITHOUT compression (to see the difference)' },
]
const DEFAULT: OrbResearchFilters = { pairs: PAIRS, ranges: RANGES, comp: 'nr4_or_tight', ml: true, tp: '2R', one_per_day: true, direction: 'all', y0: 2016, y1: 2026 }
const KEY = 'orb_research_filters'
const pfColor = (pf: number | null | undefined, n = 99) =>
  pf == null || n < 20 ? 'var(--muted)' : pf >= 1.1 ? 'var(--green)' : pf >= 1 ? 'var(--amber)' : 'var(--red)'

function saved(): OrbResearchFilters {
  try {
    const v = localStorage.getItem(KEY)
    return v ? { ...DEFAULT, ...JSON.parse(v) } : DEFAULT
  } catch {
    return DEFAULT
  }
}

function toggle(list: string[], v: string) {
  const n = list.includes(v) ? list.filter((x) => x !== v) : [...list, v]
  return n.length ? n : list
}

function StatLine({ s, label }: { s: OrbStat & { per_year?: number }; label: string }) {
  return (
    <div className="text-[12px]">
      <span className="text-[9px] tracking-[2px] text-muted">{label} </span>
      PF <b style={{ color: pfColor(s.pf, s.n) }}>{s.pf ?? '—'}</b> · {s.n} trades{s.per_year != null ? ` (${s.per_year}/yr)` : ''} · win {s.win ?? '—'}% ·{' '}
      <b style={{ color: s.r >= 0 ? 'var(--green)' : 'var(--red)' }}>{s.r > 0 ? '+' : ''}{s.r}R</b>
    </div>
  )
}

function Equity({ points }: { points: [number, number][] }) {
  const box = useRef<HTMLDivElement>(null)
  const series = useRef<ISeriesApi<'Line'> | null>(null)
  const chart = useRef<IChartApi | null>(null)
  useEffect(() => {
    if (!box.current) return
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#ffffff', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", attributionLogo: false },
      grid: { vertLines: { color: 'rgba(196,208,224,0.08)' }, horzLines: { color: 'rgba(196,208,224,0.08)' } },
      rightPriceScale: { borderColor: 'rgba(154,174,198,0.16)' },
      timeScale: { borderColor: 'rgba(154,174,198,0.16)' },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY), priceFormatter: (v: number) => `${v.toFixed(1)}R` },
    })
    series.current = c.addSeries(LineSeries, { color: '#00e5ff', lineWidth: 2 })
    chart.current = c
    return () => { c.remove(); series.current = null; chart.current = null }
  }, [])
  useEffect(() => {
    const seen = new Set<number>()
    series.current?.setData(points.filter(([t]) => (seen.has(t) ? false : (seen.add(t), true))).map(([t, v]) => ({ time: t as UTCTimestamp, value: v })))
    chart.current?.timeScale().fitContent()
  }, [points])
  return <div ref={box} className="chart-surface h-[230px] w-full" />
}

export default function OrbResearchPanel({ connected }: { connected: boolean }) {
  const [f, setF] = useState<OrbResearchFilters>(saved)
  const [rep, setRep] = useState<OrbResearchReport | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [doneKey, setDoneKey] = useState('')
  const fkey = JSON.stringify(f)
  const busy = connected && doneKey !== fkey

  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(f)) } catch { /* yaad nahi rahega, bas */ }
    if (!connected) return
    let alive = true
    const k = JSON.stringify(f)
    api.orbResearchReport(f)
      .then((r) => { if (alive) { setRep(r); setErr(null) } })
      .catch((e: unknown) => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (alive) setDoneKey(k) })
    return () => { alive = false }
  }, [f, connected])

  const set = (p: Partial<OrbResearchFilters>) => setF((o) => ({ ...o, ...p }))
  const grid = (rg: string, p: string) => rep?.by_range_pair.find((x) => x.range === rg && x.symbol === p)
  const maxAbs = Math.max(1, ...(rep?.by_year ?? []).map((y) => Math.abs(y.r)))

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>ORB RESEARCH</span>
        <em title="Pick any combination and see the backtest at once. All numbers after costs ($5/lot + 0.2 pip on stops), tick-exact entries and exits.">
          TRY ANY COMBINATION · PF / TRADES / EQUITY AFTER COSTS {busy ? '· LOADING...' : ''}
        </em>
      </div>

      <div className="mb-2 flex flex-col gap-1.5 text-[10px]">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-[86px] text-[9px] tracking-[2px] text-muted">PAIRS</span>
          {PAIRS.map((p) => <button key={p} className={`chip ${f.pairs.includes(p) ? 'on' : ''}`} onClick={() => set({ pairs: toggle(f.pairs, p) })}>{p}</button>)}
          <span className="ml-3 w-[70px] text-[9px] tracking-[2px] text-muted">SESSIONS</span>
          {RANGES.map((r) => <button key={r} className={`chip ${f.ranges.includes(r) ? 'on' : ''}`} style={{ color: ORB_COLORS[r] }}
            onClick={() => set({ ranges: toggle(f.ranges, r) })}>{r}</button>)}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-[86px] text-[9px] tracking-[2px] text-muted">COMPRESSION</span>
          {COMP.map((c) => <button key={c.id} className={`chip ${f.comp === c.id ? 'on' : ''}`} title={c.tip} onClick={() => set({ comp: c.id })}>{c.label}</button>)}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-[86px] text-[9px] tracking-[2px] text-muted">ML</span>
          <button className={`chip ${f.ml ? 'on' : ''}`} onClick={() => set({ ml: true })} title="Only breaks with a walk-forward ML score in the top 30% (2019-2026 only)">ON (TOP 30%)</button>
          <button className={`chip ${!f.ml ? 'on' : ''}`} onClick={() => set({ ml: false })} title="No ML filter (2016-2026 available)">OFF</button>
          <span className="ml-3 text-[9px] tracking-[2px] text-muted">EXIT</span>
          {[['2R', 'TP 2R'], ['1R', 'TP 1R'], ['EOD', 'HOLD TO 16:00']].map(([id, l]) => (
            <button key={id} className={`chip ${f.tp === id ? 'on' : ''}`} onClick={() => set({ tp: id })}
              title="SL always the other side of the range; anything still open closes at 16:00 NY">{l}</button>
          ))}
          <span className="ml-3 text-[9px] tracking-[2px] text-muted">SIDE</span>
          {[['all', 'BOTH'], ['bull', 'BUY'], ['bear', 'SELL']].map(([id, l]) => (
            <button key={id} className={`chip ${f.direction === id ? 'on' : ''}`} onClick={() => set({ direction: id })}>{l}</button>
          ))}
          <button className={`chip ml-3 ${f.one_per_day ? 'on' : ''}`} onClick={() => set({ one_per_day: !f.one_per_day })}
            title="Max 1 ORB trade per pair per day (the first one wins) - live rule">1 PER PAIR / DAY</button>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-[86px] text-[9px] tracking-[2px] text-muted">YEARS</span>
          <select className="chip !py-0.5" value={f.y0} onChange={(e) => set({ y0: +e.target.value })} style={{ colorScheme: 'dark' }}>
            {Array.from({ length: 11 }, (_, i) => 2016 + i).map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
          <span className="text-muted">to</span>
          <select className="chip !py-0.5" value={f.y1} onChange={(e) => set({ y1: +e.target.value })} style={{ colorScheme: 'dark' }}>
            {Array.from({ length: 11 }, (_, i) => 2016 + i).map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
          <button className="chip ml-3" onClick={() => setF(DEFAULT)} title="Back to the live bot's rules">RESET = LIVE RULES</button>
        </div>
      </div>

      {err && <div className="mb-2 text-[11px]" style={{ color: 'var(--red)' }}>{err}</div>}
      {rep && (
        <>
          <div className="mb-2 border p-2" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
            <StatLine s={rep.total} label="YOUR FILTER" />
            <div className="text-[10px] text-muted">
              max DD {rep.total.max_dd_r}R · first half PF <b style={{ color: pfColor(rep.total.pf_first_half) }}>{rep.total.pf_first_half ?? '—'}</b> · second half PF{' '}
              <b style={{ color: pfColor(rep.total.pf_second_half) }}>{rep.total.pf_second_half ?? '—'}</b>
              {rep.filters.ml_years_note ? ` · ${rep.filters.ml_years_note}` : ''}
            </div>
            <div className="mt-1 opacity-80"><StatLine s={rep.baseline} label="BASELINE (no compression, no ML)" /></div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">PF BY SESSION x PAIR [trades]</div>
              <table className="mb-3 w-full text-[11px]">
                <thead className="text-[9px] text-muted"><tr><th className="text-left font-normal">SESSION</th>
                  {PAIRS.map((p) => <th key={p} className="text-right font-normal">{p}</th>)}</tr></thead>
                <tbody>{RANGES.map((rg) => (
                  <tr key={rg} className="border-t" style={{ borderColor: 'var(--line)' }}>
                    <td className="py-0.5" style={{ color: ORB_COLORS[rg] }}>{rg}</td>
                    {PAIRS.map((p) => {
                      const g = grid(rg, p)
                      return <td key={p} className="text-right" title={g ? `${g.n} trades · win ${g.win}% · ${g.r}R` : ''}>
                        {g ? <><b style={{ color: pfColor(g.pf, g.n) }}>{g.pf ?? '—'}</b> <span className="text-[9px] text-muted">[{g.n}]</span></> : <span className="text-muted">—</span>}
                      </td>
                    })}
                  </tr>
                ))}</tbody>
              </table>
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">BY YEAR</div>
              <table className="mb-3 w-full text-[11px]">
                <tbody>{rep.by_year.map((y) => (
                  <tr key={y.year} className="border-t" style={{ borderColor: 'var(--line)' }}>
                    <td className="py-0.5">{y.year}</td><td className="text-right">{y.n}</td>
                    <td className="text-right" style={{ color: pfColor(y.pf, y.n) }}>{y.pf ?? '—'}</td>
                    <td className="text-right" style={{ color: y.r >= 0 ? 'var(--green)' : 'var(--red)' }}>{y.r > 0 ? '+' : ''}{y.r}R</td>
                    <td className="w-[40%] pl-2"><div className="h-2" style={{ width: `${(Math.abs(y.r) / maxAbs) * 100}%`, background: y.r >= 0 ? 'var(--green)' : 'var(--red)', opacity: 0.7 }} /></td>
                  </tr>
                ))}</tbody>
              </table>
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">BY WEEKDAY</div>
              <div className="flex flex-wrap gap-3 text-[11px]">
                {rep.by_weekday.map((w) => <span key={w.weekday}>{w.weekday} <b style={{ color: pfColor(w.pf, w.n) }}>{w.pf ?? '—'}</b> <span className="text-[9px] text-muted">[{w.n}]</span></span>)}
              </div>
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">EQUITY (R, after costs)</div>
              <Equity points={rep.equity} />
              <div className="mt-2 text-[9px] text-muted">
                Grey PF = under 20 trades (mostly noise). Green = PF 1.10+, amber = 1.00-1.10, red = losing. Picking only the best cells after
                looking at them = overfitting: a rule must work in both halves and make sense before it goes live.
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
