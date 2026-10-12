import { useEffect, useState } from 'react'
import { api, type OrbBrain, type OrbStat } from '../lib/api'

/*
  Research -> ORB ML brain: model kya dekhta hai (features), score ke hisaab se jeet (walk-forward), ML vs compression.
*/

const NAMES: Record<string, string> = {
  range_atr: 'Range size vs daily ATR', act_15m: 'Market activity before the break', prevday_pos: "Entry inside yesterday's range",
  asia_pos: 'Entry inside the Asia range (London)', spread_r: 'Spread vs SL', risk_pips: 'SL size (pips)',
  break_delay_min: 'Minutes from range end to break', w_ratio: 'Range vs last 20 ranges', weekday: 'Weekday',
  h4_aligned: 'H4 trend in break direction', d1_mom_aligned: '20-day trend in break direction',
  prevday_aligned: "Yesterday's candle in break direction", symbol: 'Pair', range: 'Session (range)',
  nr3: 'NR3', nr4: 'NR4', nr5: 'NR5', nr6: 'NR6', nr7: 'NR7', inside: 'Inside day',
}
const pfColor = (pf: number | null | undefined) => (pf == null ? 'var(--muted)' : pf >= 1.1 ? 'var(--green)' : pf >= 1 ? 'var(--amber)' : 'var(--red)')

function Row({ label, s, tip }: { label: string; s: OrbStat; tip: string }) {
  return (
    <tr className="border-t" style={{ borderColor: 'var(--line)' }} title={tip}>
      <td className="py-0.5">{label}</td><td className="text-right">{s.n.toLocaleString()}</td><td className="text-right">{s.win ?? '—'}%</td>
      <td className="text-right"><b style={{ color: pfColor(s.pf) }}>{s.pf ?? '—'}</b></td>
      <td className="text-right" style={{ color: s.r >= 0 ? 'var(--green)' : 'var(--red)' }}>{s.r > 0 ? '+' : ''}{s.r}R</td>
    </tr>
  )
}

export default function OrbBrainPanel({ connected }: { connected: boolean }) {
  const [b, setB] = useState<OrbBrain | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    if (!connected) return
    api.orbResearchBrain().then(setB).catch((e: unknown) => setErr(e instanceof Error ? e.message : String(e)))
  }, [connected])

  const maxShare = Math.max(0.01, ...(b?.importance ?? []).map((x) => x.share))
  const maxAbs = Math.max(1, ...(b?.buckets ?? []).map((x) => Math.abs(x.r)))

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>ORB ML BRAIN</span>
        <em title={b?.note}>WHAT THE ORB MODEL LOOKS AT · WIN RATE BY SCORE · ML vs COMPRESSION</em>
      </div>
      {err && <div className="text-[11px]" style={{ color: 'var(--red)' }}>{err}</div>}
      {b && (
        <>
          <div className="mb-2 text-[10px] text-muted">
            Trained {b.meta.trained_at.slice(0, 10)} on {b.meta.rows.toLocaleString()} breakouts (data to {b.meta.data_to}) · keeps top {Math.round(b.meta.keep * 100)}%
            (score ≥ {Math.round(b.meta.threshold * 100)}%) · label: {b.meta.label}
          </div>
          <div className="grid gap-4 xl:grid-cols-3">
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">WHAT IT LOOKS AT (share of model gain)</div>
              {b.importance.slice(0, 12).map((x) => (
                <div key={x.feature} className="mb-0.5 text-[10px]" title={x.feature}>
                  <div className="flex justify-between"><span>{NAMES[x.feature] ?? x.feature}</span><span className="text-muted">{(x.share * 100).toFixed(1)}%</span></div>
                  <div className="h-1.5" style={{ width: `${(x.share / maxShare) * 100}%`, background: 'var(--cyan)', opacity: 0.7 }} />
                </div>
              ))}
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted" title="All breakouts 2019-2026 split into 10 equal groups by ML score (walk-forward). If the model works, higher groups do better.">
                SCORE GROUPS (all breaks 2019-26, TP 2R, net)
              </div>
              <table className="w-full text-[10px]">
                <thead className="text-[9px] text-muted"><tr>
                  <th className="text-left font-normal">SCORE</th><th className="text-right font-normal">N</th><th className="text-right font-normal">WIN</th>
                  <th className="text-right font-normal">PF</th><th className="w-[35%] font-normal" />
                </tr></thead>
                <tbody>{b.buckets.map((x) => (
                  <tr key={x.bucket} className="border-t" style={{ borderColor: 'var(--line)', background: x.score_from >= b.meta.threshold * 0.999 ? 'rgba(0,229,255,0.05)' : undefined }}>
                    <td className="py-0.5">{Math.round(x.score_from * 100)}-{Math.round(x.score_to * 100)}%</td>
                    <td className="text-right">{x.n}</td><td className="text-right">{x.win}%</td>
                    <td className="text-right" style={{ color: pfColor(x.pf) }}>{x.pf ?? '—'}</td>
                    <td className="pl-2"><div className="h-1.5" style={{ width: `${(Math.abs(x.r) / maxAbs) * 100}%`, background: x.r >= 0 ? 'var(--green)' : 'var(--red)', opacity: 0.7 }} /></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[9px] tracking-[2px] text-muted">ML vs COMPRESSION (2019-26, TP 2R, net)</div>
              <table className="w-full text-[10px]">
                <thead className="text-[9px] text-muted"><tr>
                  <th className="text-left font-normal">GROUP</th><th className="text-right font-normal">TRADES</th><th className="text-right font-normal">WIN</th>
                  <th className="text-right font-normal">PF</th><th className="text-right font-normal">R</th>
                </tr></thead>
                <tbody>
                  <Row label="All breaks" s={b.split.all_breaks} tip="Every breakout, no filter" />
                  <Row label="Compression only" s={b.split.comp_no_ml} tip="NR4 or tight, no ML" />
                  <Row label="ML only (top 30%)" s={b.split.ml_keep} tip="ML top 30%, compressed or not" />
                  <Row label="ML dropped" s={b.split.ml_drop} tip="Breaks the ML would skip" />
                  <Row label="ML + compression (LIVE)" s={b.split.ml_keep_comp} tip="The live bot rule (before the 1-per-day limit)" />
                  <Row label="ML but NOT compressed" s={b.split.ml_keep_nocomp} tip="ML likes it but the market was not compressed" />
                </tbody>
              </table>
              <div className="mt-2 text-[9px] text-muted">
                Neither filter alone is enough: compression alone ~1.0, ML alone ~1.07. Together they give the edge.
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
