r"""
MT5 CONNECTION CHECK (sirf PADHTA hai, koi order NAHI bhejta).

Kya check karta hai:
  1. Python laptop pe khule MT5 desktop se connect ho pa raha hai?
  2. Kaunsa account login hai (DEMO hona chahiye), balance, leverage
  3. "Allow algorithmic trading" on hai?
  4. Hamare 3 pairs ke live price, spread (pips), lot rules (min lot, step, contract size)
  5. Aakhri 3 M5 candles
  6. Broker SERVER TIME vs UTC ka farak (MT5 candles server time mein deta hai, humara sab data UTC mein
     hai -> ye farak bot ko pata hona zaroori hai)

Pehle: MT5 desktop khula + demo account login + pip install MetaTrader5
Chalao: python src\live\mt5_check.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

PAIRS = ["EURUSD", "USDJPY", "GBPUSD"]


def find_symbol(mt5, name: str):
    """Kuch brokers naam ke saath suffix lagate hain (EURUSD.r, EURUSDm) -> dhoondh lo."""
    if mt5.symbol_info(name) is not None:
        return name
    cands = [s.name for s in (mt5.symbols_get(f"*{name}*") or [])]
    return sorted(cands, key=len)[0] if cands else None


def main() -> None:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        sys.exit("MetaTrader5 package nahi hai. Chalao: pip install MetaTrader5")

    if not mt5.initialize():
        sys.exit(f"MT5 se connect nahi hua: {mt5.last_error()}\n"
                 "-> MT5 desktop khula hai? Account login hai? Tools > Options > Expert Advisors > Allow algorithmic trading?")

    try:
        term = mt5.terminal_info()
        acc = mt5.account_info()
        print("=== TERMINAL ===")
        print(f"  {term.name} | build {mt5.version()[1]} | connected to server: {term.connected}")
        print(f"  algo trading allowed: {term.trade_allowed}" + ("" if term.trade_allowed else
              "   <-- MT5 toolbar mein 'Algo Trading' button ON karo"))

        print("\n=== ACCOUNT ===")
        demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        print(f"  login {acc.login} | server {acc.server} | {'DEMO' if demo else '!!! REAL / CONTEST !!!'}")
        print(f"  balance {acc.balance:,.2f} {acc.currency} | equity {acc.equity:,.2f} | leverage 1:{acc.leverage}")
        print(f"  hedging allowed: {acc.margin_mode == mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING}")
        if not demo:
            print("  WARNING: ye DEMO account nahi hai. Bot ke tests sirf demo pe karenge.")

        print("\n=== PAIRS ===")
        offsets = []
        for p in PAIRS:
            sym = find_symbol(mt5, p)
            if sym is None:
                print(f"  {p}: is broker pe nahi mila")
                continue
            mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
            tick = mt5.symbol_info_tick(sym)
            pip = 0.01 if "JPY" in p else 0.0001
            spread = (tick.ask - tick.bid) / pip
            print(f"  {sym:10s} bid {tick.bid:.{info.digits}f}  ask {tick.ask:.{info.digits}f}  spread {spread:.1f} pip | "
                  f"min lot {info.volume_min} step {info.volume_step} | contract {info.trade_contract_size:,.0f}")
            offsets.append(tick.time)
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 3)
            if rates is not None:
                for r in rates:
                    t = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    print(f"      M5 {t} (server time)  O {r['open']} H {r['high']} L {r['low']} C {r['close']}")

        print("\n=== SERVER TIME ===")
        if offsets:
            now = datetime.now(timezone.utc).timestamp()
            off_h = round((max(offsets) - now) / 3600)
            print(f"  broker server time = UTC {off_h:+d}h  (last tick {now - max(offsets) + off_h * 3600:.0f}s ago)")
            print("  (market band ho, jaise weekend, toh last tick purana hoga aur ye number galat ho sakta hai)")
        print("\nOK: connection kaam kar raha hai. Koi order nahi bheja gaya.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
