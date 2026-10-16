import { useEffect, useState } from 'react'
import { api, type WeeklyReport, type WeeklyBox } from '../lib/api'
import { ZONE_TYPE_TAG } from '../lib/zonesPrimitive'

/*
  WEEKEND REPORT (bot writes it every Saturday NY, src/live/weekly_report.py):
  what the bot did, what YOU did (AVOID / TAKE / EXIT / CLOSE) and whose decision was right, in R.
*/
const R = (v: number | null | undefined) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(2)}R`)
const col = (v: number | null | undefined, invert = false) =>
  v == null || v === 0 ? 'var(--text)' : (v > 0) !== invert ? 'var(--green)' : 'var(--red)'

function Row({ name, tip, b, invert = false }: { name: string; tip: string; b?: WeeklyBox | null; invert?: boolean }) {
  return (
    <tr title={tip}>
      <td>{name}</td>
      <td className="text-right">{b?.n ?? 0}</td>
      <td className="text-right" style={{ color: col(b?.R, invert) }}>{R(b?.R)}</td>
      <td className="text-right">{b?.pf == null ? '—' : b.pf === Infinity ? '∞' : b.pf}</td>
    </tr>
  )
}

function Scope({ title, z }: { title: string; z: WeeklyReport['zone_week'] }) {
  const e = z?.your_exit_changes
  const e2 = z?.your_trade_exit_changes
  const c = z?.your_closes
  return (
    <div className="min-w-[260px] flex-1">
      <div className="mb-1 text-[9px] tracking-[2px] text-muted">{title}</div>
      <table className="w-full text-[11px]">
        <thead className="text-[9px] text-muted">
          <tr><th className="text-left font-normal">ZONES</th><th className="text-right font-normal">N</th><th className="text-right font-normal">R</th><th className="text-right font-normal">PF</th></tr>
        </thead>
        <tbody>
          <Row name="Bot took" tip="Zones the bot traded (R after costs, with the exit used)" b={z?.bot_taken} />
          <Row name="You avoided" tip="R the bot WOULD have made on the zones you avoided. Negative (red) = good for you: your AVOID saved that much." b={z?.you_avoided} invert />
          <Row name="You took" tip="Zones you TOOK although the bot skipped them (ML low / value area / quiet)" b={z?.you_taken} />
          <Row name="ML low (skipped)" tip="R the bot would have made if it had taken its ML-low zones. Negative = the ML was right to skip." b={z?.ml_low_skipped} invert />
        </tbody>
      </table>
      <div className="mt-1 flex flex-col gap-0.5 text-[10px] text-muted">
        <span title="Exit you set before the trade vs the exit brain's pick, on the zones that were touched">
          Your exit (before trade): {e?.n ?? 0} · <b style={{ color: col(e?.R_gained) }}>{R(e?.R_gained)}</b> vs brain{e?.n ? ` (${e.right} right / ${e.wrong} wrong)` : ''}
        </span>
        <span title="Exit you switched during an open trade vs the exit brain's pick on the same zone">
          Exit switched in trade: {e2?.matched ?? 0} · <b style={{ color: col(e2?.R_gained) }}>{R(e2?.R_gained)}</b> vs brain{e2?.matched ? ` (${e2.right} right / ${e2.wrong} wrong)` : ''}
        </span>
        <span title="R at your manual close minus the R the trade's own exit gave">
          Your manual closes: {c?.n ?? 0} · <b style={{ color: col(c?.R_gained) }}>{R(c?.R_gained)}</b> vs letting them run
        </span>
      </div>
    </div>
  )
}

export default function WeeklyReportPanel({ connected }: { connected: boolean }) {
  const [rep, setRep] = useState<WeeklyReport | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    if (!connected) return
    let alive = true
    const load = () => api.liveWeekly().then((r) => { if (alive) { setRep(r); setErr(null) } }).catch((e) => alive && setErr(String(e)))
    load()
    const id = window.setInterval(load, 10 * 60_000)
    return () => { alive = false; window.clearInterval(id) }
  }, [connected])

  return (
    <div className="panel">
      <div className="mb-2 flex items-baseline gap-3">
        <h2 className="text-[11px] tracking-[3px]">WEEKEND REPORT</h2>
        <span className="text-[10px] text-muted">
          {rep?.from_ && rep.to ? `${rep.from_.slice(0, 10)} → ${rep.to.slice(0, 10)} · written by the bot every Saturday (NY)` : ''}
        </span>
      </div>
      {err ? (
        <div className="text-[11px] text-muted">Could not load the report: {err}</div>
      ) : !rep || rep.status === 'none' || rep.error ? (
        <div className="text-[11px] text-muted">{rep?.error ?? 'No report yet. The bot writes the first one on Saturday (New York time).'}</div>
      ) : (
        <>
          <div className="mb-2 flex flex-wrap gap-4 text-[11px]">
            <span>ZONE BOT: {rep.zone_account?.n ?? 0} trades · <b style={{ color: col(rep.zone_account?.R) }}>{R(rep.zone_account?.R)}</b> · ${(rep.zone_account?.pnl_usd ?? 0).toFixed(2)}</span>
            {(['normal', 'confluence', 'h4'] as const).map((k) => {
              const b = rep.zone_by_type_week?.[k]
              if (!b || !b.n) return null
              return (
                <span key={k} style={{ color: ZONE_TYPE_TAG[k].color }} title="Zone bot trades of this zone type closed this week">
                  {ZONE_TYPE_TAG[k].text}: {b.n} · {R(b.R)} · PF {b.pf ?? '—'} · ${b.pnl_usd.toFixed(2)}
                </span>
              )
            })}
            <span>ORB: {rep.orb_account?.n ?? 0} trades · <b style={{ color: col(rep.orb_account?.R) }}>{R(rep.orb_account?.R)}</b> · ${(rep.orb_account?.pnl_usd ?? 0).toFixed(2)}{rep.orb_account?.manual_closes ? ` · ${rep.orb_account.manual_closes} closed by you` : ''}</span>
            {Object.entries(rep.real_week ?? {}).map(([k, v]) => (
              <span key={k} title="Real DEMO orders on MT5 vs the same shadow trades">
                REAL {k.toUpperCase().replace('_', ' ')}: {v.n} · R {R(v.real_R)} vs shadow {R(v.shadow_R)} · ${v.net_usd.toFixed(2)}{v.avg_slip_pips != null ? ` · slip ${v.avg_slip_pips} pips` : ''}
              </span>
            ))}
            <span className="text-muted">Your actions this week: {Object.entries(rep.your_actions_week ?? {}).map(([k, v]) => `${k} ${v}`).join(', ') || 'none'}</span>
          </div>
          <div className="flex flex-wrap gap-6">
            <Scope title="THIS WEEK" z={rep.zone_week} />
            <Scope title="SINCE THE START" z={rep.zone_all_time} />
          </div>
          <div className="mt-2 text-[9px] text-muted">Judge your decisions only after 100+ of them: a few weeks can be luck. Monthly training uses all of this.</div>
        </>
      )}
    </div>
  )
}
