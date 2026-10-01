r"""
Risk Agent v1: REALISTIC portfolio simulation.

Ab tak ke results "har trade ₹100 risk" wale the (R mein). Asli account aisa nahi chalta:
  - Paisa (equity) upar-neeche hota hai, lot size usi hisaab se banta hai
  - Lot size 0.01 se chhota nahi ho sakta (chhote account mein kai trades skip honge)
  - Broker commission leta hai, SL pe slippage hoti hai
  - Ek saath kai trades khule ho sakte hain (sab mein USD hai = ek hi bet)
  - Bura din ho toh bot ko khud ruk jaana chahiye

Ye file yahi sab simulate karti hai, ML ke OUT-OF-SAMPLE trades pe (2019+, walk-forward):
  Entry filter : M15 zone, pehla touch, ML top 30% (green)
  Exit         : Exit Brain ka choice (1:1 ya trailing)

RISK RULES (RiskConfig):
  risk_pct          : har trade mein equity ka kitna % risk (default 0.25%)
  min_sl_pips       : isse chhote SL wale trades skip (default 0: cost-aware ML khud handle karta hai)
  max_open          : ek saath max kitne trades
  max_per_symbol    : ek pair pe max kitne
  max_usd_same_side : USD ki ek hi direction mein max kitne (EURUSD buy = USD sell, USDJPY buy = USD buy)
  daily_loss_pct    : aaj ka realized loss itna ho jaaye toh aaj naya trade nahi (NY din)
  brake_dd_pct      : DRAWDOWN BRAKE - account peak se itna % neeche -> risk aadha (default 5%), naye peak pe normal
  kill_dd_pct       : KILL SWITCH - peak se itna % neeche -> naye trades band + alert (default 9%, prop limit 10%)
  commission_per_lot: round-trip commission USD per 1.00 lot (Funding Pips $5, ECN brokers ~$6-7)
  slippage_pips     : SL / trailing / breakeven exit pe itne pips extra nuksaan

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\risk\portfolio_sim.py
    python src\risk\portfolio_sim.py --capital 100 --risk 1

Pehle ML train hona chahiye (ml_scores.parquet), dashboard ka RETRAIN ya walkforward.py.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research.management import POLICIES, run_trade_full  # noqa: E402

DATA = ROOT / "data" / "processed"
RESEARCH = ROOT / "data" / "research"
NY = "America/New_York"
HOLD_BARS = 3 * 288
POL = {p.name: p for p in POLICIES}
EXIT_POLICY = {"tp1": POL["tp1"], "trail": POL["trail1"]}
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCAD": 0.0001, "AUDUSD": 0.0001}
LOT = 100_000   # 1.00 lot = 1 lakh units
KILL_PAUSE_DAYS = 7


@dataclass
class RiskConfig:
    capital: float = 1000.0
    risk_pct: float = 0.25       # normal risk per trade (% of equity)
    # DRAWDOWN BRAKE: account peak se itna % neeche -> risk x factor (0.25% -> 0.125%), naye peak pe wapas normal
    brake_dd_pct: float = 5.0
    brake_factor: float = 0.5
    # KILL SWITCH: peak se itna % neeche -> naye trades BAND (live bot tumhe alert karega, tum review karke resume karoge).
    # Backtest mein: KILL_PAUSE_DAYS din ruk ke half risk pe resume (maan ke chalte hain ki tumne review karke chalu kiya).
    kill_dd_pct: float = 9.0
    min_sl_pips: float = 0.0     # cost-aware ML khud chhote-SL wale mehenge trades avoid karta hai; purane ML ke saath 6 tha
    commission_per_lot: float = 5.0
    max_open: int = 3
    max_per_symbol: int = 1
    max_usd_same_side: int = 2
    daily_loss_pct: float = 3.0
    slippage_pips: float = 0.2
    min_lot: float = 0.01
    lot_step: float = 0.01


# ----------------------------------------------------------------------
# 1) Candidates: ML ne jo trades chune, unka exit time + R (M5 pe dobara chala ke)
# ----------------------------------------------------------------------
def load_candidates(symbols: list[str] | None = None, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """scores na diya ho toh data/research/ml_scores.parquet se padhta hai."""
    if scores is None:
        f = RESEARCH / "ml_scores.parquet"
        if not f.exists():
            raise SystemExit("ml_scores.parquet nahi mila. Pehle ML train karo (dashboard RETRAIN ya walkforward.py).")
        scores = pd.read_parquet(f)
    s = scores.copy()
    s["symbol"] = s["symbol"].astype(str)
    s["zone_tf"] = s["zone_tf"].astype(str)
    s["direction"] = s["direction"].astype(str)
    if "exit_ml" not in s:
        raise SystemExit("ml_scores mein exit_ml nahi hai. ML dobara train karo (naya walkforward.py).")
    s = s[(s["zone_tf"] == "M15") & (s["touch_number"] == 1) & s["keep_0.3"]]
    if symbols:
        s = s[s["symbol"].isin(symbols)]

    rows = []
    for sym, g in s.groupby("symbol"):
        m5 = pd.read_parquet(DATA / f"{sym}_M5.parquet", columns=["high", "low", "close", "spread"])
        hi, lo, cl, sp = (m5[c].to_numpy() for c in ["high", "low", "close", "spread"])
        idx = m5.index
        for r in g.itertuples(index=False):
            f0 = int(idx.searchsorted(r.fill_time))
            g1 = min(f0 + HOLD_BARS, len(m5))
            bull = r.direction == "bull"
            risk = abs(r.entry - r.sl)
            rr, why, k = run_trade_full(EXIT_POLICY[r.exit_ml], bull, r.entry, risk,
                                        hi[f0:g1], lo[f0:g1], cl[f0:g1], sp[f0:g1])
            rows.append(dict(symbol=sym, direction=r.direction, fill_time=r.fill_time,
                             exit_time=idx[f0 + k] + pd.Timedelta(minutes=5), entry=r.entry, risk=risk,
                             exit=r.exit_ml, r=rr, reason=why, score=r.score))
    return pd.DataFrame(rows).sort_values("fill_time").reset_index(drop=True)


def usd_side(symbol: str, direction: str) -> int:
    """+1 = trade USD kharid raha hai, -1 = USD bech raha hai."""
    usd_base = symbol.startswith("USD")
    buy = direction == "bull"
    return 1 if (usd_base == buy) else -1


def risk_usd_per_unit(symbol: str, entry: float, risk: float) -> float:
    """1 unit (1 lot = 1 lakh units) pe SL lage toh kitne USD jaayenge."""
    if symbol.endswith("USD"):          # EURUSD, GBPUSD, AUDUSD: quote currency USD hai
        return risk
    return risk / entry                 # USDJPY, USDCAD: quote JPY/CAD -> USD mein badlo


# ----------------------------------------------------------------------
# 2) Simulation: time ke hisaab se ek ek trade, asli account jaisa
# ----------------------------------------------------------------------
def simulate(cands: pd.DataFrame, cfg: RiskConfig) -> dict:
    equity = cfg.capital
    peak = equity                      # ab tak ka sabse upar ka balance (brake / kill switch isi se naapte hain)
    paused_until = None                # kill switch ke baad kab tak naye trades band
    kill_peak = None                   # kis peak pe kill switch laga (naya peak bane tabhi dobara lag sakta hai)
    kill_events: list[dict] = []
    open_pos: list[dict] = []
    closed: list[dict] = []
    skipped: dict[str, int] = {}
    day_pnl: dict = {}
    curve = [(cands["fill_time"].iloc[0] - pd.Timedelta(days=1), equity)]

    def close_until(t):
        nonlocal equity, peak
        for p in sorted([p for p in open_pos if p["exit_time"] <= t], key=lambda p: p["exit_time"]):
            equity += p["pnl"]
            peak = max(peak, equity)
            day = p["exit_time"].tz_convert(NY).date()
            day_pnl[day] = day_pnl.get(day, 0.0) + p["pnl"]
            closed.append(p)
            open_pos.remove(p)
            curve.append((p["exit_time"], equity))

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for c in cands.itertuples(index=False):
        close_until(c.fill_time)
        if equity <= 0:
            skip("account blown")
            continue
        dd = (peak - equity) / peak * 100
        # KILL SWITCH: bahut gira -> ruk jao (live mein tum review karke resume karoge)
        if dd >= cfg.kill_dd_pct and kill_peak != peak:
            kill_peak = peak
            paused_until = c.fill_time + pd.Timedelta(days=KILL_PAUSE_DAYS)
            kill_events.append({"time": str(c.fill_time.date()), "drawdown_pct": round(dd, 1)})
        if paused_until is not None and c.fill_time < paused_until:
            skip("kill switch pause")
            continue
        today = c.fill_time.tz_convert(NY).date()
        if day_pnl.get(today, 0.0) <= -cfg.daily_loss_pct / 100 * equity:
            skip("daily loss limit")
            continue
        if len(open_pos) >= cfg.max_open:
            skip("max open trades")
            continue
        if sum(p["symbol"] == c.symbol for p in open_pos) >= cfg.max_per_symbol:
            skip("already in this pair")
            continue
        if c.risk / PIP[c.symbol] < cfg.min_sl_pips:
            skip("SL too small for costs")
            continue
        side = usd_side(c.symbol, c.direction)
        if sum(p["usd_side"] == side for p in open_pos) >= cfg.max_usd_same_side:
            skip("too much USD one way")
            continue

        # DRAWDOWN BRAKE: peak se brake_dd_pct neeche -> risk aadha
        braked = dd >= cfg.brake_dd_pct
        risk_pct = cfg.risk_pct * (cfg.brake_factor if braked else 1.0)

        # Lot size: risk amount / (SL pe per-unit nuksaan), 0.01 ke steps mein NEECHE round
        per_unit = risk_usd_per_unit(c.symbol, c.entry, c.risk)
        units = (equity * risk_pct / 100) / per_unit
        lots = np.floor(units / LOT / cfg.lot_step) * cfg.lot_step
        if lots < cfg.min_lot:
            skip("lot below 0.01 (account too small)")
            continue
        risk_usd = lots * LOT * per_unit
        slip_usd = 0.0
        if c.reason in ("sl", "trail", "be"):
            slip_price = cfg.slippage_pips * PIP[c.symbol]
            slip_usd = lots * LOT * (slip_price if c.symbol.endswith("USD") else slip_price / c.entry)
        pnl = c.r * risk_usd - cfg.commission_per_lot * lots - slip_usd
        open_pos.append(dict(symbol=c.symbol, direction=c.direction, usd_side=side, fill_time=c.fill_time,
                             exit_time=c.exit_time, lots=round(float(lots), 2), risk_usd=risk_usd,
                             r=c.r, pnl=pnl, exit=c.exit, reason=c.reason, braked=braked))
    close_until(pd.Timestamp.max.tz_localize("UTC"))
    res = report(closed, curve, skipped, cfg, len(cands))
    res["braked_trades_pct"] = round(100 * float(np.mean([p["braked"] for p in closed])), 1) if closed else 0.0
    res["kill_switch_events"] = kill_events
    return res


# ----------------------------------------------------------------------
# 3) Report
# ----------------------------------------------------------------------
def report(closed: list[dict], curve: list, skipped: dict, cfg: RiskConfig, n_cands: int) -> dict:
    t = pd.DataFrame(closed)
    eq = pd.Series([v for _, v in curve], index=pd.DatetimeIndex([k for k, _ in curve]))
    peak = eq.cummax()
    dd_pct = ((peak - eq) / peak).max() * 100
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    final = float(eq.iloc[-1])
    cagr = ((final / cfg.capital) ** (1 / years) - 1) * 100 if final > 0 else -100.0

    monthly = eq.resample("ME").last().ffill()
    m_ret = monthly.pct_change().dropna() * 100
    yearly = eq.resample("YE").last().ffill()
    y_start = pd.concat([pd.Series([cfg.capital], index=[yearly.index[0] - pd.offsets.YearEnd()]), yearly]).shift(1).iloc[1:]
    y_ret = ((yearly / y_start - 1) * 100).round(1)

    gw, gl = t.loc[t["pnl"] > 0, "pnl"].sum(), -t.loc[t["pnl"] < 0, "pnl"].sum()
    streak = longest = 0
    for p in t.sort_values("exit_time")["pnl"]:
        streak = streak + 1 if p < 0 else 0
        longest = max(longest, streak)

    daily = eq.resample("D").last().ffill()
    return dict(
        config=asdict(cfg),
        start=str(eq.index[0].date()), end=str(eq.index[-1].date()), years=round(years, 2),
        candidates=n_cands, trades=len(t), skipped=skipped,
        trades_per_month=round(len(t) / (years * 12), 1),
        start_equity=cfg.capital, final_equity=round(final, 2),
        total_return_pct=round((final / cfg.capital - 1) * 100, 1),
        cagr_pct=round(cagr, 1), max_drawdown_pct=round(float(dd_pct), 1),
        win_pct=round(100 * float((t["pnl"] > 0).mean()), 1),
        profit_factor=round(float(gw / gl), 2) if gl else None,
        commission_usd=round(float((cfg.commission_per_lot * t["lots"]).sum()), 2),
        avg_lots=round(float(t["lots"].mean()), 2),
        losing_months_pct=round(100 * float((m_ret < 0).mean()), 1),
        worst_month_pct=round(float(m_ret.min()), 1), best_month_pct=round(float(m_ret.max()), 1),
        longest_losing_streak=int(longest),
        yearly_return_pct={str(k.year): float(v) for k, v in y_ret.items()},
        by_symbol={s: dict(trades=int(len(g)), pnl=round(float(g["pnl"].sum()), 2)) for s, g in t.groupby("symbol")},
        equity_curve=[{"time": int(k.timestamp()), "value": round(float(v), 2)} for k, v in daily.items()],
    )


def run(capital: float = 1000, risk_pct: float = 0.25, commission: float = 5.0, min_sl_pips: float = 0.0,
        symbols=None, cands=None) -> dict:
    cfg = RiskConfig(capital=capital, risk_pct=risk_pct, commission_per_lot=commission, min_sl_pips=min_sl_pips)
    cands = load_candidates(symbols) if cands is None else cands
    if symbols:
        cands = cands[cands["symbol"].isin(symbols)].reset_index(drop=True)
    return simulate(cands, cfg)


def save(res: dict) -> Path:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    f = RESEARCH / "portfolio_sim.json"
    f.write_text(json.dumps(res, indent=2, default=str))
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Risk Agent: realistic portfolio simulation")
    ap.add_argument("--capital", type=float, default=1000)
    ap.add_argument("--risk", type=float, default=0.25, help="risk per trade, %% of equity")
    ap.add_argument("--commission", type=float, default=5.0, help="USD per 1.00 lot round trip")
    ap.add_argument("--min-sl", type=float, default=0.0, help="skip trades with SL smaller than this (pips)")
    a = ap.parse_args()

    print("Loading ML-selected trades (out-of-sample) and replaying exits on M5...")
    cands = load_candidates()
    res = run(a.capital, a.risk, a.commission, a.min_sl, cands=cands)
    show = {k: v for k, v in res.items() if k not in ("equity_curve", "config")}
    print(json.dumps(show, indent=2, default=str))
    save(res)
    print("\nsaved -> data/research/portfolio_sim.json")


if __name__ == "__main__":
    main()
