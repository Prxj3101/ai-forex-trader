import { useEffect, useRef } from 'react'
import type { MLReport, RiskReport } from '../lib/api'
import { rgba, useTheme } from '../lib/theme'

type Props = { report: MLReport | null; accountReport: RiskReport | null; training: boolean; onTrain: () => void; connected: boolean }

// Feature ke code-names ko dashboard ke liye simple English mein
const FEATURE_LABELS: Record<string, string> = {
  spread_r: 'Spread vs stop size',
  risk_atr: 'Stop size (ATR)',
  bars_since_created: 'Zone age at fill',
  range_pos_d1: 'Premium/discount (D1)',
  range_pos_h4: 'Premium/discount (H4)',
  touch_number: 'Touch number',
  pullback_atr: 'Pullback depth',
  fill_hour_ny: 'Fill hour (NY)',
  spread_atr: 'Spread (ATR)',
  disp_h4: 'Displacement (H4 ATR)',
  disp_h1: 'Displacement (H1 ATR)',
  disp_atr: 'Displacement (ATR)',
  zone_height_atr: 'Zone height',
  break_margin_atr: 'Break margin',
  session_fill: 'Session at fill',
  killzone: 'Killzone',
  has_fvg: 'Fair value gap',
  approach_1h_atr: 'Approach speed (1h)',
  approach_3h_atr: 'Approach speed (3h)',
  approach_push: 'Approach pressure',
  followthrough_atr: 'Follow-through after BOS',
  impulse_candles: 'Impulse candles',
  impulse_body_pct: 'Impulse body strength',
  disp_d1: 'Displacement (D1 ATR)',
  trend_tf_aligned: 'Local trend (M15)',
  trend_h1_aligned: 'Local trend (H1)',
  cost_r: 'Cost vs stop size',
  sl_pips: 'Stop size (pips)',
}

const BUCKET_COLORS = ['#ff426e', '#ff8a3d', '#ffd000', '#9be15d', '#00ff9c']

// ML model ka panel: trained hai toh asli walk-forward results, warna NOT TRAINED
export default function MLBrain({ report, accountReport, training, onTrain, connected }: Props) {
  const { theme } = useTheme()
  const ref = useRef<HTMLCanvasElement>(null)
  const level = useRef(0)   // 0 = not trained (dim), 1 = trained, 2 = training (bahut active)
  const trained = !!report

  useEffect(() => {
    level.current = training ? 2 : trained ? 1 : 0
  }, [training, trained])

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const layers = [6, 8, 8, 4, 1]
    let pulses: { li: number; a: number; b: number; p: number }[] = []
    let raf = 0

    const draw = (t: number) => {
      const r = cv.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      cv.width = r.width * dpr
      cv.height = r.height * dpr
      const ctx = cv.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const W = r.width
      const H = r.height
      const lv = level.current
      const pos = layers.map((n, li) =>
        Array.from({ length: n }, (_, i) => [12 + (li * (W - 24)) / (layers.length - 1), 10 + ((i + 0.5) * (H - 20)) / n]),
      )
      ctx.lineWidth = 0.6
      for (let li = 0; li < layers.length - 1; li++) {
        for (const a of pos[li]) {
          for (const b of pos[li + 1]) {
            ctx.strokeStyle = rgba(theme, lv ? 0.38 : 0.1)
            ctx.beginPath()
            ctx.moveTo(a[0], a[1])
            ctx.lineTo(b[0], b[1])
            ctx.stroke()
          }
        }
      }
      if (Math.random() < [0.08, 0.3, 0.9][lv]) {
        const li = Math.floor(Math.random() * (layers.length - 1))
        pulses.push({ li, a: Math.floor(Math.random() * layers[li]), b: Math.floor(Math.random() * layers[li + 1]), p: 0 })
      }
      pulses = pulses.filter((q) => q.p <= 1)
      for (const q of pulses) {
        const A = pos[q.li][q.a]
        const B = pos[q.li + 1][q.b]
        ctx.fillStyle = lv ? 'rgba(255,196,0,0.95)' : 'rgba(255,196,0,0.5)'
        ctx.beginPath()
        ctx.arc(A[0] + (B[0] - A[0]) * q.p, A[1] + (B[1] - A[1]) * q.p, lv ? 2 : 1.6, 0, Math.PI * 2)
        ctx.fill()
        q.p += lv === 2 ? 0.06 : 0.03
      }
      pos.forEach((l, li) =>
        l.forEach(([x, y]) => {
          const glow = 0.5 + 0.5 * Math.sin(t / 400 + x + y)
          const out = li === layers.length - 1
          ctx.fillStyle = out ? (lv ? '#ffd000' : '#9aa8bd') : rgba(theme, (lv ? 0.85 : 0.4) + (lv ? 0.15 : 0.4) * glow)
          ctx.beginPath()
          ctx.arc(x, y, out ? 5 : 2.4, 0, Math.PI * 2)
          ctx.fill()
        }),
      )
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [theme])

  const all = report?.summary.find((s) => s.keep === 1)
  const bot = report?.bot ?? null
  const combo = accountReport?.config.combo_enabled ? accountReport : null
  const lastYear = report ? Math.max(...report.by_year.map((y) => y.year)) : 0
  const gauge = combo?.win_pct ?? bot?.win_pct ?? report?.score_buckets[report.score_buckets.length - 1]?.real_win_pct ?? 0
  const netTag = report?.net ? 'NET' : 'GROSS'
  const status = training ? 'TRAINING' : trained ? 'TRAINED' : 'NOT TRAINED'
  const statusColor = training ? 'var(--cyan)' : trained ? 'var(--green)' : 'var(--amber)'

  return (
    <div className="panel">
      <div className="ptitle"><span>ML BRAIN</span><em>ZONE SCORING</em></div>

      <div className="flex items-center gap-3">
        <div className="relative h-[110px] w-[110px] flex-none" title={combo ? 'Win rate of executed combo account trades after position limits, news blocks and trading costs' : 'Win rate of strategy-selected trades after costs, on years the model never saw'}>
          <div className="absolute inset-0 rounded-full border border-dashed" style={{ borderColor: rgba(theme, 0.35), animation: `spin ${training ? 3 : 12}s linear infinite` }} />
          <div
            className="absolute inset-[6px] rounded-full"
            style={{
              background: trained
                ? `conic-gradient(#ffd000 ${gauge * 3.6}deg, rgba(255,255,255,0.06) 0)`
                : 'rgba(255,255,255,0.06)',
              mask: 'radial-gradient(circle, transparent 36px, #000 37px)',
              WebkitMask: 'radial-gradient(circle, transparent 36px, #000 37px)',
              filter: trained ? 'drop-shadow(0 0 6px rgba(255,196,0,0.6))' : 'none',
            }}
          />
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <b className="font-display text-2xl" style={{ color: trained ? '#ffd000' : 'var(--muted)' }}>
              {trained ? `${gauge.toFixed(1)}%` : '--'}
            </b>
            <small className="text-center text-[8px] leading-3 tracking-wider text-muted">{combo ? 'COMBO WIN' : 'STRATEGY WIN'}<br />RATE ({combo ? 'NET' : netTag})</small>
          </div>
        </div>
        <div className="text-[11px] leading-6 text-muted">
          <div>STATUS <b className="font-normal" style={{ color: statusColor }}>{status}</b></div>
          <div>TRADES LEARNED <b className="font-normal text-[var(--text)]">{report ? report.n_trades.toLocaleString() : '--'}</b></div>
          <div>TEST YEARS <b className="font-normal text-[var(--text)]">{report ? `${report.first_test_year}-${lastYear}` : '--'}</b></div>
          <div>MODEL <b className="font-normal text-[var(--text)]">LightGBM</b></div>
          {report?.net && (
            <div title="Every result in this panel already has these costs subtracted">
              COSTS <b className="font-normal text-[var(--text)]">${report.commission}/lot + {report.slippage_pips} pip</b>
            </div>
          )}
        </div>
      </div>

      <canvas ref={ref} className="my-2 block h-[90px] w-full" title="Neural activity: how the model combines features into a decision" />

      {combo && (
        <div className="mb-3 border px-2 py-2 text-[10px]" style={{ borderColor: 'var(--cyan)', background: 'var(--panel2)' }}>
          <div className="mb-1 font-display tracking-wider" style={{ color: 'var(--cyan)' }}>COMBO ACCOUNT · NET</div>
          <div className="mb-2 leading-4 text-muted">
            BASE 0.5% / 0.25% · REDUCE AT {combo.config.combo_reduce_dd_pct}% DD · RECOVER AT {combo.config.combo_recover_dd_pct}% DD
          </div>
          <div className="grid grid-cols-2 gap-x-2 gap-y-2">
            <div><small className="block text-muted">ACCOUNT PF</small><b className="font-display text-base" style={{ color: 'var(--green)' }}>{combo.profit_factor == null ? '—' : combo.profit_factor.toFixed(2)}</b></div>
            <div><small className="block text-muted">CAGR / YEAR</small><b className="font-display text-base" style={{ color: 'var(--green)' }}>{combo.cagr_pct.toFixed(1)}%</b></div>
            <div><small className="block text-muted">MAX CLOSED DD</small><b className="font-display text-base" style={{ color: 'var(--amber)' }}>−{combo.max_drawdown_pct.toFixed(1)}%</b></div>
            <div><small className="block text-muted">ACCOUNT TRADES</small><b className="font-display text-base">{combo.trades.toLocaleString()}</b></div>
          </div>
          <div className="mt-2 text-[9px] text-muted">HISTORICAL SIMULATION · {combo.start} TO {combo.end}</div>
          <div className="mt-1 text-[9px] leading-4 text-muted">Account results include risk sizing, news blocks and position limits. Strategy stats below measure signals in R.</div>
        </div>
      )}

      {report && all ? (
        <>
          <div className="mb-2 grid grid-cols-2 gap-2 text-center">
            <div
              className="border px-2 py-1.5"
              style={{ borderColor: 'var(--line)' }}
              title="If the bot traded EVERY zone (all timeframes, every touch, trailing exit). Unseen years only."
            >
              <small className="block text-[9px] tracking-wider text-muted">EVERY ZONE PF ({netTag})</small>
              <b className="font-display text-lg font-medium" style={{ color: all.pf_trail < 1 ? 'var(--red)' : 'var(--text)' }}>
                {all.pf_trail.toFixed(2)}
              </b>
              <small className="block text-[9px] text-muted">{all.trades.toLocaleString()} trades</small>
            </div>
            <div
              className="border px-2 py-1.5"
              style={{ borderColor: 'rgba(0,255,136,0.5)', boxShadow: '0 0 12px rgba(0,255,136,0.15) inset' }}
              title="What the bot actually does: M15 zone, first touch, ML top 30%, value area filter, exit brain picks 1:1 or trailing. Unseen years only."
            >
              <small className="block text-[9px] tracking-wider text-muted">STRATEGY PF ({netTag})</small>
              <b className="font-display text-lg font-medium" style={{ color: 'var(--green)' }}>
                {bot ? bot.pf.toFixed(2) : '--'}
              </b>
              <small className="block text-[9px] text-muted">
                {bot ? `${bot.trades.toLocaleString()} trades · ${bot.losing_years} losing yrs` : 'retrain to update'}
              </small>
            </div>
          </div>
          {report?.va_filter?.on && (
            <div
              className="mb-1 text-[10px] text-muted"
              title="After the ML picks a trade, the bot skips it if the entry is on the expensive side of the last 24h volume profile value area (buy near or above the value area high, sell near or below the value area low). Threshold set from pre-2019 data only."
            >
              VALUE AREA FILTER <b className="font-normal" style={{ color: 'var(--green)' }}>ON</b> · skipped{' '}
              {report.va_filter.removed.toLocaleString()} expensive entries · PF without filter {report.va_filter.pf_without.toFixed(2)}
            </div>
          )}
          {report?.act_filter?.on && (
            <div
              className="mb-1 text-[10px] text-muted"
              title="After the ML (and value area filter) pick a trade, the bot skips it if the market was quieter than usual right before the touch: tick volume of the last 15 min below 0.9x the 24h average. Threshold set from pre-2021 data only."
            >
              ACTIVITY FILTER <b className="font-normal" style={{ color: 'var(--green)' }}>ON</b> · skipped{' '}
              {report.act_filter.removed.toLocaleString()} quiet-market touches · PF without filter {report.act_filter.pf_without.toFixed(2)}
            </div>
          )}
          {bot && (
            <div className="mb-2 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted" title="Strategy-selected trades per pair in net R; account sizing is not applied here">
              {Object.entries(bot.by_symbol).map(([s, v]) => (
                <span key={s}>
                  {s} <b className="font-normal" style={{ color: v.pf >= 1 ? 'var(--green)' : 'var(--red)' }}>STRATEGY PF {v.pf.toFixed(2)}</b>
                </span>
              ))}
            </div>
          )}

          {report.exits?.length > 0 && (
            <>
              <div className="mb-1 text-[9px] tracking-[2px] text-muted" title="Same trades (M15 first touch, ML top 30%), different exits, unseen years">
                STRATEGY EXITS (SAME TRADES, {netTag})
              </div>
              <div className="mb-2 flex flex-col gap-0.5 text-[10px]">
                {report.exits.map((e) => {
                  const best = e.exit === 'ml'
                  return (
                    <div
                      key={e.exit}
                      className="flex items-center justify-between px-1.5 py-0.5"
                      style={best ? { background: 'rgba(255,196,0,0.1)', border: '1px solid rgba(255,196,0,0.5)' } : { border: '1px solid transparent' }}
                      title={`${e.trades.toLocaleString()} trades, ${e.losing_years} losing years`}
                    >
                      <span className={best ? '' : 'text-muted'} style={best ? { color: '#ffd000' } : undefined}>{e.label}</span>
                      <span>
                        PF <b className="font-normal" style={{ color: best ? '#ffd000' : 'var(--text)' }}>{e.pf.toFixed(2)}</b>
                        <span className="text-muted"> · DD {e.max_dd_r.toFixed(0)}R</span>
                      </span>
                    </div>
                  )
                })}
              </div>
            </>
          )}

          <div className="mb-1 text-[9px] tracking-[2px] text-muted" title="Every zone, split by ML score. Shows the model ranks trades: higher score, better result.">
            SCORE GROUPS (EVERY ZONE, {netTag})
          </div>
          <div className="mb-2 flex flex-col gap-1">
            {report.score_buckets.map((b, i) => (
              <div key={b.bucket} className="flex items-center gap-2 text-[10px]" title={`${b.trades.toLocaleString()} trades, avg score ${(b.avg_score * 100).toFixed(0)}%`}>
                <span className="w-[62px] flex-none text-muted">{b.bucket.toUpperCase()}</span>
                <div className="h-2 flex-1 bg-white/5">
                  <div className="h-full" style={{ width: `${Math.max(4, Math.min(100, (b.pf - 0.8) / 0.6 * 100))}%`, background: BUCKET_COLORS[i], boxShadow: `0 0 6px ${BUCKET_COLORS[i]}` }} />
                </div>
                <span className="w-[74px] flex-none text-right" style={{ color: BUCKET_COLORS[i] }}>
                  {b.real_win_pct.toFixed(0)}% · PF {b.pf.toFixed(2)}
                </span>
              </div>
            ))}
          </div>

          <div className="mb-1 text-[9px] tracking-[2px] text-muted">WHAT THE MODEL LOOKS AT</div>
          <div className="mb-2 flex flex-col gap-0.5">
            {report.importance.slice(0, 5).map((f) => (
              <div key={f.feature} className="flex items-center gap-2 text-[10px]">
                <span className="w-[130px] flex-none truncate">{FEATURE_LABELS[f.feature] ?? f.feature}</span>
                <div className="h-1.5 flex-1 bg-white/5">
                  <div className="h-full" style={{ width: `${Math.min(100, f.pct * 5)}%`, background: 'var(--cyan)' }} />
                </div>
                <span className="w-8 flex-none text-right text-muted">{f.pct.toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="mb-2 border px-3 py-2 text-center text-[11px] leading-5" style={{ borderColor: 'var(--line)', color: 'var(--muted)' }}>
          {training ? 'Training in progress: one model per test year...' : 'No trained model yet. Build dataset v2, then train.'}
        </div>
      )}

      <button
        className="chip on w-full py-1.5"
        disabled={training || !connected}
        onClick={onTrain}
        title="Walk-forward: for every year, train only on earlier years and test on that year"
      >
        {training ? 'TRAINING...' : trained ? 'RETRAIN MODEL' : 'TRAIN MODEL'}
      </button>
    </div>
  )
}
