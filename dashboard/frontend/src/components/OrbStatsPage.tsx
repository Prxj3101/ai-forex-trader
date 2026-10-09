import { ColorType, LineSeries, createChart, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from 'lightweight-charts'
import { useEffect, useRef, useState } from 'react'
import { api, type OrbBacktest, type OrbGroupStat, type OrbLive } from '../lib/api'
import { ORB_COLORS } from '../lib/orbPrimitive'
import { NY, fmtTs } from '../lib/time'

/*
  ORB STATS page: backtest (walk-forward 2019-26, model never saw the year it traded) vs live shadow,
  session (range) x pair breakdown, saal-wise, equity curve. Isse dikhega ki live backtest jaisa chal raha hai ya nahi
  aur kaunsa session / pair sach mein kaam karta hai (jaise GBP London, JPY Asia).
*/

const PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
const RANGES = ['ASIA', 'LON30', 'LON60'] as const
const RANGE_TIP: Record<string, string> = {
  ASIA: 'Asia range 19:00-02:00 NY, breakout until 11:00',
  LON30: 'London first 30 min 03:00-03:30 NY, breakout until 11:00',
  LON60: 'London first hour 03:00-04:00 NY, breakout until 11:00',
}
const pfColor = (pf: number | null | undefined, n = 99) =>
  pf == null || n < 5 ? 'var(--muted)' : pf >= 1.1 ? 'var(--green)' : pf >= 1 ? 'var(--amber)' : 'var(--red)'

function Grid({ rows, title, tip }: { rows: OrbGroupStat[]; title: string; tip: string }) {
  const get = (rg: string, s: string) => rows.find((r) => r.range === rg && r.symbol === s)
  return (
    <div className="min-w-0">
      <div className="mb-1 text-[9px] tracking-[2px] text-muted" title={tip}>{title}</div>
      <table className="w-full text-[11px]">
        <thead className="text-[9px] text-muted"><tr>
          <th className="text-left font-normal">RANGE</th>
          {PAIRS.map((p) => <th key={p} className="text-right font-normal">{p}</th>)}
        </tr></thead>
        <tbody>{RANGES.map((rg) => (
          <tr key={rg} className="border-t" style={{ borderColor: 'var(--line)' }}>
            <td className="py-1" style={{ color: ORB_COLORS[rg] }} title={RANGE_TIP[rg]}>{rg}</td>
            {PAIRS.map((p) => {
              const g = get(rg, p)
              return (
                <td key={p} className="text-right" title={g ? `${g.n} trades · win ${g.win}% · ${g.r > 0 ? '+' : ''}${g.r}R` : 'no trades'}>
                  {g ? <><b style={{ color: pfColor(g.pf, g.n) }}>{g.pf ?? '—'}</b> <span className="text-[9px] text-muted">[{g.n}]</span></> : <span className="text-muted">—</span>}
                </td>
              )
            })}
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

function Summary({ label, g, extra }: { label: string; g?: OrbGroupStat; extra?: string }) {
  return (
    <div className="border p-2" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
      <div className="text-[9px] tracking-[2px] text-muted">{label}</div>
      {g ? (
        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[12px]">
          <span title="Profit factor after costs">PF <b style={{ color: pfColor(g.pf, g.n) }}>{g.pf ?? '—'}</b></span>
          <span>TRADES <b>{g.n}</b></span>
          <span>WIN <b>{g.win}%</b></span>
          <span>R <b style={{ color: g.r >= 0 ? 'var(--green)' : 'var(--red)' }}>{g.r > 0 ? '+' : ''}{g.r}</b></span>
          {extra && <span className="text-muted">{extra}</span>}
        </div>
      ) : <div className="mt-1 text-[11px] text-muted">No trades yet.</div>}
    </div>
  )
}

function EquityChart({ points, unit }: { points: [number, number][]; unit: string }) {
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<ISeriesApi<'Line'> | null>(null)
  useEffect(() => {
    if (!box.current) return
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#ffffff', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", attributionLogo: false },
      grid: { vertLines: { color: 'rgba(196,208,224,0.08)' }, horzLines: { color: 'rgba(196,208,224,0.08)' } },
      rightPriceScale: { borderColor: 'rgba(154,174,198,0.16)' },
      timeScale: { borderColor: 'rgba(154,174,198,0.16)' },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number, NY), priceFormatter: (v: number) => `${v.toFixed(unit === 'R' ? 1 : 0)}${unit === 'R' ? 'R' : ''}` },
    })
    series.current = c.addSeries(LineSeries, { color: '#00e5ff', lineWidth: 2 })
    chart.current = c
    return () => { c.remove(); chart.current = null; series.current = null }
  }, [unit])
  useEffect(() => {
    const seen = new Set<number>()
    const data = points.filter(([t]) => (seen.has(t) ? false : (seen.add(t), true))).map(([t, v]) => ({ time: t as UTCTimestamp, value: v }))
    series.current?.setData(data)
    chart.current?.timeScale().fitContent()
  }, [points])
  return <div ref={box} className="chart-surface h-[220px] w-full" />
}

export default function OrbStatsPage({ connected }: { connected: boolean }) {
  const [bt, setBt] = useState<OrbBacktest | null>(null)
  const [live, setLive] = useState<OrbLive | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!connected) return
    api.orbBacktest().then(setBt).catch((e: unknown) => setErr(e instanceof Error ? e.message : String(e)))
    let alive = true
    const load = () => api.liveOrb().then((x) => { if (alive) setLive(x) }).catch(() => undefined)
    void load()
    const id = window.setInterval(load, 30_000)
    return () => { alive = false; window.clearInterval(id) }
  }, [connected])

  const ls = live?.status === 'ok' ? live.live_stats : undefined
  const maxAbsR = Math.max(1, ...(bt?.by_year ?? []).map((y) => Math.abs(y.r)))

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>ORB STATS</span>
        <em title="Backtest numbers come from the walk-forward test (the model never saw the year it traded). Live numbers come from the shadow bot on the VPS.">
          BACKTEST (2019-2026, OUT OF SAMPLE) VS LIVE SHADOW · BY SESSION AND PAIR
        </em>
      </div>
      {err && <div className="mb-2 text-[11px]" style={{ color: 'var(--red)' }}>Backtest stats: {err}</div>}
      {bt?.model && (
        <div className="mb-2 text-[10px] text-muted">
          MODEL trained {bt.model.trained_at.slice(0, 10)} on {bt.model.rows.toLocaleString()} breakouts (data to {bt.model.data_to}) · ML keeps top {Math.round(bt.model.keep * 100)}% (score ≥ {Math.round(bt.model.threshold * 100)}%) ·
          compression = NR4 or range ≤ {Object.entries(bt.model.tight_q30).map(([k, v]) => `${k} ${v.toFixed(2)}x`).join(' / ')} of normal · TP {bt.model.tp_r}R · max 1 ORB per pair per day
        </div>
      )}

      <div className="mb-3 grid gap-2 md:grid-cols-2">
        <Summary label="BACKTEST 2019-2026 (walk-forward, after costs)" g={bt?.total} extra={bt ? `max DD ${bt.max_dd_r}R` : undefined} />
        <Summary label={`LIVE SHADOW${live?.account ? ` · equity $${live.account.equity.toLocaleString()}` : ''}`} g={ls?.total} />
      </div>

      <div className="mb-3 grid gap-4 md:grid-cols-2">
        {bt && <Grid rows={bt.by_range_pair} title="BACKTEST PF BY SESSION x PAIR [trades]" tip="Green = PF 1.10 or better, amber = 1.00-1.10, red = losing" />}
        <Grid rows={ls?.by_range_pair ?? []} title="LIVE PF BY SESSION x PAIR [trades]" tip="Fills up as the shadow bot trades. Under ~30 trades per cell the numbers are mostly noise." />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">BACKTEST BY YEAR</div>
          <table className="w-full text-[11px]">
            <thead className="text-[9px] text-muted"><tr>
              <th className="text-left font-normal">YEAR</th><th className="text-right font-normal">TRADES</th><th className="text-right font-normal">WIN</th>
              <th className="text-right font-normal">PF</th><th className="text-right font-normal">R</th><th className="w-[40%] font-normal" />
            </tr></thead>
            <tbody>{(bt?.by_year ?? []).map((y) => (
              <tr key={y.year} className="border-t" style={{ borderColor: 'var(--line)' }}>
                <td className="py-0.5">{y.year}</td><td className="text-right">{y.n}</td><td className="text-right">{y.win}%</td>
                <td className="text-right" style={{ color: pfColor(y.pf) }}>{y.pf ?? '—'}</td>
                <td className="text-right" style={{ color: y.r >= 0 ? 'var(--green)' : 'var(--red)' }}>{y.r > 0 ? '+' : ''}{y.r}</td>
                <td className="pl-2"><div className="h-2" style={{ width: `${(Math.abs(y.r) / maxAbsR) * 100}%`, background: y.r >= 0 ? 'var(--green)' : 'var(--red)', opacity: 0.7 }} /></td>
              </tr>
            ))}</tbody>
          </table>
          <div className="mt-2 grid grid-cols-2 gap-3 text-[11px]">
            <div>
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">BACKTEST BY SESSION</div>
              {(bt?.by_range ?? []).map((g) => <div key={g.range} className="flex justify-between"><span style={{ color: ORB_COLORS[g.range ?? ''] }}>{g.range}</span><span style={{ color: pfColor(g.pf) }}>{g.pf} <span className="text-muted">[{g.n}]</span></span></div>)}
            </div>
            <div>
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">BACKTEST BY PAIR</div>
              {(bt?.by_symbol ?? []).map((g) => <div key={g.symbol} className="flex justify-between"><span>{g.symbol}</span><span style={{ color: pfColor(g.pf) }}>{g.pf} <span className="text-muted">[{g.n}]</span></span></div>)}
            </div>
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">BACKTEST EQUITY (R, after costs)</div>
          {bt && <EquityChart points={bt.equity} unit="R" />}
          {(live?.equity_curve?.length ?? 0) > 1 && (
            <>
              <div className="mb-1 mt-3 text-[9px] tracking-[2px] text-muted">LIVE SHADOW EQUITY ($)</div>
              <EquityChart points={live!.equity_curve!} unit="$" />
            </>
          )}
        </div>
      </div>
    </div>
  )
}
