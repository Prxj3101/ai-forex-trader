import { AreaSeries, ColorType, createChart, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from 'lightweight-charts'
import { motion } from 'framer-motion'
import { useEffect, useRef } from 'react'
import type { RiskReport } from '../lib/api'
import { rgba, useTheme } from '../lib/theme'
import { fmtTs } from '../lib/time'

/*
  Risk Agent ka account simulation: "agar bot 2019 se is account pe chalta toh kya hota?"
  Wahi numbers jo chat mein batate hain (asli lot size, commission, slippage, risk rules ke saath).
*/

const usd = (v: number) => `$${Math.round(v).toLocaleString()}`

function EquityCurve({ data }: { data: RiskReport['equity_curve'] }) {
  const { theme } = useTheme()
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<ISeriesApi<'Area'> | null>(null)

  useEffect(() => {
    if (!box.current) return
    const c = createChart(box.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#c3cfdf', fontFamily: "'Share Tech Mono', monospace", attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: 'rgba(255,255,255,0.06)' } },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, tickMarkFormatter: (t: Time) => new Date((t as number) * 1000).getUTCFullYear().toString() },
      localization: { timeFormatter: (t: Time) => fmtTs(t as number), priceFormatter: (p: number) => usd(p) },
      handleScroll: false,
      handleScale: false,
    })
    series.current = c.addSeries(AreaSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true })
    chart.current = c
    return () => c.remove()
  }, [])

  useEffect(() => {
    series.current?.applyOptions({ lineColor: theme.accent, topColor: rgba(theme, 0.35), bottomColor: rgba(theme, 0.02) })
  }, [theme])

  useEffect(() => {
    series.current?.setData(data.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })))
    chart.current?.timeScale().fitContent()
  }, [data])

  return <div ref={box} className="h-[230px] w-full" />
}

function Stat({ label, value, color, tip }: { label: string; value: string; color?: string; tip: string }) {
  return (
    <div title={tip} className="cursor-help">
      <small className="block text-[9px] tracking-wider text-muted">{label}</small>
      <b className="font-display text-[15px] font-medium" style={{ color: color ?? 'var(--text)' }}>{value}</b>
    </div>
  )
}

export default function AccountPanel({ report }: { report: RiskReport | null }) {
  if (!report) {
    return (
      <div className="panel">
        <div className="ptitle"><span>ACCOUNT SIMULATION</span><em>RISK AGENT</em></div>
        <div className="py-6 text-center text-[11px] text-muted">
          No simulation yet. Press RETRAIN MODEL in the ML Brain panel: the Risk Agent runs automatically after training.
        </div>
      </div>
    )
  }
  const c = report.config
  const ap = report.account_profile
  const years = Object.entries(report.yearly_return_pct)
  const maxAbs = Math.max(...years.map(([, v]) => Math.abs(v)), 1)

  return (
    <div className="panel">
      <div className="ptitle flex-wrap">
        <span>ACCOUNT SIMULATION</span>
        <em title="Real lot sizes, commission, slippage and risk rules, on trades the ML picked for years it never saw">
          RISK AGENT // {report.start} TO {report.end}
        </em>
      </div>

      {ap && (
        <div
          className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 border px-2.5 py-1.5 text-[10px] tracking-wider"
          style={{ borderColor: ap.news_rules ? 'var(--amber)' : 'var(--line)', background: 'var(--panel2)' }}
        >
          <b className="font-display font-medium" style={{ color: ap.news_rules ? 'var(--amber)' : 'var(--green)' }}>
            ACCOUNT: {ap.label}
          </b>
          {ap.news_rules ? (
            <>
              <span title="No new trades or pending orders inside this window around high-impact news on the pair's currencies">
                NO TRADES {ap.news_before_min} MIN BEFORE TO {ap.news_after_min} MIN AFTER NEWS
              </span>
              <span title="Speech window runs until the speech ends (assumed 60 minutes long)">
                SPEECHES {ap.speech_before_min} MIN BEFORE TO {ap.speech_after_min} MIN AFTER END
              </span>
              <span title={`Trades opened less than ${ap.exempt_hours} hours before the news are ${ap.pre_news === 'close' ? 'closed before the window' : 'held (profit removed if they exit inside the window)'}`}>
                {ap.pre_news === 'close' ? `CLOSE TRADES < ${ap.exempt_hours}H OLD BEFORE NEWS` : 'HOLD THROUGH NEWS'}
              </span>
              <span className="text-muted" title={`News calendar in this backtest ends ${ap.calendar_end}; trades after that date had no news rules applied`}>
                · {ap.blocked_fills?.toLocaleString()} ENTRIES BLOCKED · {ap.closed_before_news} CLOSED EARLY
                {ap.profit_removed ? ` · ${ap.profit_removed} PROFITS REMOVED` : ''} · CALENDAR TO {ap.calendar_end}
              </span>
            </>
          ) : ap.news_block ? (
            <span
              className="text-muted"
              title={`No new trades from ${ap.block_before_min} min before to ${ap.block_after_min} min after high-impact news on the pair's currencies (spreads and slippage are worse there than the backtest can see). Open trades are not touched. News calendar in this backtest ends ${ap.calendar_end}.`}
            >
              NO PROP NEWS RULES
            </span>
          ) : (
            <span className="text-muted">NO NEWS RESTRICTIONS</span>
          )}
          {ap.news_block && (
            <span title={`No new trades from ${ap.block_before_min} min before to ${ap.block_after_min} min after high-impact news on the pair's currencies. Open trades are not touched.`}>
              · NEWS BLOCK -{ap.block_before_min} TO +{ap.block_after_min} MIN
              {!ap.news_rules && ` (${ap.blocked_fills?.toLocaleString()} ENTRIES BLOCKED · CALENDAR TO ${ap.calendar_end})`}
            </span>
          )}
        </div>
      )}

      <div className="mb-2 text-[10px] tracking-wider text-muted">
        START {usd(c.capital)} · RISK {c.risk_pct}% PER TRADE · COSTS ${c.commission_per_lot}/LOT + {c.slippage_pips} PIP · MAX {c.max_open} OPEN{c.daily_risk_budget_pct === undefined ? ` · DAILY STOP ${c.daily_loss_pct}%` : ''}
        {c.brake_dd_pct !== undefined && (
          <span title="When the balance is this far below its peak, risk per trade is cut until a new peak">
            {' '}· BRAKE AT -{c.brake_dd_pct}% (RISK DROPS TO {+(c.risk_pct * (c.brake_factor ?? 1)).toFixed(3)}%)
          </span>
        )}
        {report.va_filter?.on && (
          <span title="Trades the ML picked but the bot skipped because the entry was on the expensive side of the last 24h value area">
            {' '}· VALUE AREA FILTER ON ({report.va_filter.skipped.toLocaleString()} SKIPPED)
          </span>
        )}
        {report.act_filter?.on && (
          <span title="Trades the bot skipped because the market was quieter than usual right before the touch">
            {' '}· ACTIVITY FILTER ON ({report.act_filter.skipped.toLocaleString()} SKIPPED)
          </span>
        )}
        {!!c.house_money_pct && (
          <span title={`Risk per trade grows with this month's profit: base ${c.risk_pct}% + ${c.house_money_pct}% of the month's profit, never above ${c.risk_cap_pct}%. Resets every month. The drawdown brake overrides it.`}>
            {' '}· HOUSE MONEY +{c.house_money_pct}% OF MONTH PROFIT (MAX {c.risk_cap_pct}%)
          </span>
        )}
        {c.daily_risk_budget_pct !== undefined && (
          <span title={`No new trades after a ${c.daily_loss_pct}% closed loss today. A new trade only opens if today's loss + every open stop loss + the new trade stays under ${c.daily_risk_budget_pct}% (prop firm daily limit 3%). Day = broker day, 5 PM New York to 5 PM.`}>
            {' '}· DAILY STOP {c.daily_loss_pct}% · DAILY RISK BUDGET {c.daily_risk_budget_pct}%
          </span>
        )}
        {c.kill_dd_pct !== undefined && (
          <span title="At this drawdown the bot stops opening trades and alerts you. In this simulation it resumes after a 7 day pause.">
            {' '}· KILL SWITCH AT -{c.kill_dd_pct}%
          </span>
        )}
      </div>

      <div className="grid gap-3 xl:grid-cols-[260px_minmax(0,1fr)_220px]">
        <div className="grid grid-cols-2 content-start gap-x-3 gap-y-2.5 border p-2.5" style={{ borderColor: 'var(--line)', background: 'var(--panel2)' }}>
          <Stat label="FINAL BALANCE" value={usd(report.final_equity)} color="var(--green)" tip={`Started with ${usd(report.start_equity)}`} />
          <Stat label="RETURN / YEAR" value={`${report.cagr_pct}%`} color={report.cagr_pct > 0 ? 'var(--green)' : 'var(--red)'} tip="Compounded yearly growth (CAGR)" />
          <Stat label="MAX DRAWDOWN" value={`-${report.max_drawdown_pct}%`} color={report.max_drawdown_pct > 10 ? 'var(--red)' : 'var(--amber)'} tip="Biggest fall from a peak. Prop firms usually stop you at 10%." />
          <Stat label="WORST MONTH" value={`${report.worst_month_pct}%`} color="var(--red)" tip={`Best month: +${report.best_month_pct}%`} />
          {report.worst_day_pct != null && (
            <Stat label="WORST DAY" value={`${report.worst_day_pct}%`} color={report.worst_day_pct < -3 ? 'var(--red)' : 'var(--amber)'} tip="Biggest loss in one broker day (5 PM to 5 PM New York). Funding Pips daily limit is 3%." />
          )}
          {report.risk_used && (
            <Stat label="AVG RISK / TRADE" value={`${report.risk_used.avg_pct}%`} tip={`Max ${report.risk_used.max_pct}% · ${report.risk_used.boosted_trades_pct}% of trades used extra risk from this month's profit`} />
          )}
          <Stat label="PROFIT FACTOR" value={report.profit_factor.toFixed(2)} color={report.profit_factor >= 1.1 ? 'var(--green)' : 'var(--amber)'} tip="Money won / money lost, after costs" />
          <Stat label="WIN RATE" value={`${report.win_pct}%`} tip={`Longest losing streak: ${report.longest_losing_streak} trades`} />
          <Stat label="TRADES / MONTH" value={`${report.trades_per_month}`} tip={`${report.trades.toLocaleString()} trades in total`} />
          <Stat label="LOSING MONTHS" value={`${report.losing_months_pct}%`} tip="Share of months that ended in a loss" />
          <Stat label="COMMISSION PAID" value={usd(report.commission_usd)} color="var(--muted)" tip="Total broker commission over the whole period" />
          {report.braked_trades_pct !== undefined && (
            <Stat label="BRAKE ACTIVE" value={`${report.braked_trades_pct}%`} color="var(--amber)" tip="Share of trades taken at reduced risk because the account was in a drawdown" />
          )}
          {report.kill_switch_events !== undefined && (
            <Stat
              label="KILL SWITCH HITS"
              value={`${report.kill_switch_events.length}`}
              color={report.kill_switch_events.length ? 'var(--red)' : 'var(--green)'}
              tip={report.kill_switch_events.map((e) => `${e.time}: -${e.drawdown_pct}%`).join('\n') || 'Never triggered'}
            />
          )}
          <Stat
            label="SKIPPED TRADES"
            value={Object.values(report.skipped).reduce((a, b) => a + b, 0).toLocaleString()}
            color="var(--muted)"
            tip={Object.entries(report.skipped).map(([k, v]) => `${k}: ${v}`).join('\n') || 'None'}
          />
        </div>

        <div className="min-w-0">
          <div className="mb-1 text-[9px] tracking-[2px] text-muted">BALANCE OVER TIME</div>
          <EquityCurve data={report.equity_curve} />
        </div>

        <div>
          <div className="mb-1.5 text-[9px] tracking-[2px] text-muted">RETURN BY YEAR</div>
          {years.map(([y, v]) => (
            <div key={y} className="mb-1 grid grid-cols-[38px_1fr_52px] items-center gap-2 text-[11px]">
              <span className="text-muted">{y}</span>
              <div className="relative h-2.5" style={{ background: 'rgba(255,255,255,0.05)' }}>
                <motion.i
                  className="absolute inset-y-0 left-0 block"
                  initial={{ width: 0 }}
                  animate={{ width: `${(Math.abs(v) / maxAbs) * 100}%` }}
                  transition={{ duration: 0.6 }}
                  style={{ background: v >= 0 ? 'var(--green)' : 'var(--red)', boxShadow: `0 0 8px ${v >= 0 ? 'var(--green)' : 'var(--red)'}` }}
                />
              </div>
              <b className="text-right font-normal" style={{ color: v >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {v > 0 ? '+' : ''}{v}%
              </b>
            </div>
          ))}
          <div className="mt-2 text-[9px] leading-4 text-muted">
            Backtest on years the model never saw. Live results will be lower (fills, news spikes). Demo test comes next.
          </div>
        </div>
      </div>
    </div>
  )
}
