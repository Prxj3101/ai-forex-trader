import type { RealSummary } from '../lib/api'

/*
  REAL vs SHADOW (data/live/real_trades.csv, src/live/execution.py): same trade, shadow (ticks, no broker) vs the real
  DEMO order on MT5. Shows how much the broker (slippage, spread, fills) changes the result.
*/
const R = (v: number | null | undefined, d = 2) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}`)
const col = (v: number | null | undefined) => (v == null || v === 0 ? 'var(--text)' : v > 0 ? 'var(--green)' : 'var(--red)')
const NAME: Record<string, string> = { zone: 'ZONE BOT', orb: 'ORB' }

export default function RealVsShadowPanel({ real }: { real?: RealSummary | null }) {
  const strats = Object.entries(real?.strategies ?? {})
  return (
    <div className="panel">
      <div className="mb-2 flex items-baseline gap-3">
        <h2 className="text-[11px] tracking-[3px]">REAL vs SHADOW</h2>
        <span className="text-[10px] text-muted">real DEMO orders on MT5 next to the shadow trades · slippage + = real fill worse</span>
      </div>
      {strats.length === 0 ? (
        <div className="text-[11px] text-muted">No closed real trades yet. Turn ON real DEMO orders in LIVE MARKET (ZONE BOT / ORB).</div>
      ) : (
        <>
          <div className="mb-3 flex flex-wrap gap-6 text-[11px]">
            {strats.map(([k, s]) => (
              <div key={k} className="min-w-[230px]">
                <div className="mb-1 text-[9px] tracking-[2px] text-muted">{NAME[k] ?? k.toUpperCase()}</div>
                <div>{s.trades} trades · {s.wins} won · net <b style={{ color: col(s.net_usd) }}>${R(s.net_usd)}</b> <span className="text-muted">(commission ${s.commission_usd.toFixed(2)})</span></div>
                <div title="Total R of the same trades: real fills vs shadow">
                  R real <b style={{ color: col(s.real_r) }}>{R(s.real_r)}</b> vs shadow <b style={{ color: col(s.shadow_r) }}>{R(s.shadow_r)}</b>
                  {s.r_gap != null && <span className="text-muted"> · {R(s.r_gap, 3)}R per trade</span>}
                </div>
                <div className="text-muted">avg entry slippage {s.avg_entry_slip_pips == null ? '—' : `${R(s.avg_entry_slip_pips)} pips`}</div>
              </div>
            ))}
          </div>
          <div className="max-h-[32vh] overflow-y-auto">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 text-[9px] text-muted" style={{ background: 'var(--panel)' }}>
                <tr>
                  <th className="text-left font-normal">STRATEGY</th><th className="text-left font-normal">PAIR</th>
                  <th className="text-right font-normal">SHADOW ENTRY</th><th className="text-right font-normal">REAL ENTRY</th>
                  <th className="text-right font-normal">SLIP</th><th className="text-right font-normal">SHADOW R</th>
                  <th className="text-right font-normal">REAL R</th><th className="text-right font-normal">EXIT</th><th className="text-right font-normal">NET $</th>
                </tr>
              </thead>
              <tbody>
                {(real?.trades ?? []).map((t, i) => (
                  <tr key={i} title={t.note ?? ''}>
                    <td>{NAME[t.strategy] ?? t.strategy}</td>
                    <td>{t.symbol} <span style={{ color: t.direction === 'bull' ? 'var(--green)' : 'var(--red)' }}>{t.direction === 'bull' ? 'BUY' : 'SELL'}</span></td>
                    <td className="text-right">{t.shadow_entry ?? '—'}</td>
                    <td className="text-right">{t.real_entry ?? '—'}</td>
                    <td className="text-right" style={{ color: t.entry_slip_pips != null && t.entry_slip_pips > 0 ? 'var(--red)' : undefined }}>{t.entry_slip_pips == null ? '—' : R(t.entry_slip_pips, 1)}</td>
                    <td className="text-right" style={{ color: col(t.shadow_r) }}>{R(t.shadow_r)}</td>
                    <td className="text-right" style={{ color: col(t.real_r) }}>{R(t.real_r)}</td>
                    <td className="text-right text-muted">{t.real_exit_reason ?? '—'}</td>
                    <td className="text-right" style={{ color: col(t.net_usd) }}>{t.net_usd == null ? '—' : `$${R(t.net_usd)}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
