r"""
REAL DEMO EXECUTION (Phase B, 15 Oct 2026): shadow bot ke trades Exness MT5 DEMO pe ASLI orders ke roop mein.
Shadow (nakli) trades saath mein chalte rehte hain -> har trade pe shadow vs real ka farak (slippage, fill, P&L).

SAFETY
  - Sirf DEMO account (account_info().trade_mode == DEMO), warna ek bhi order nahi.
  - Switch: dashboard (control.json "exec": {"zone": bool, "orb": bool}), shuru mein dono OFF.
    OFF karne pe: naye orders band + zone ke pending orders cancel. Khule real trades apne exit tak manage hote rehte hain.
  - Har order pe magic number: ZONE 1001, ORB 2002. Bot sirf apne magic wale orders / positions chhuta hai.
  - MAX_LOTS: isse bada lot kabhi nahi.
ZONE BOT  armed order (READY zone, current candle) -> MT5 BUY/SELL LIMIT (entry, SL, TP 1R for 1:1, no TP for trail).
          Zone READY nahi raha / AVOID / candle badli aur order dobara armed nahi -> cancel.
          Fill -> shadow trade se jodo (same symbol + zone + direction). Trail / exit switch -> SL / TP modify.
          Shadow trade band (timeout / manual / trail) aur real abhi khula -> market pe band.
          Real fill hua par shadow ne trade nahi liya (risk rule at fill) -> real turant band (note "shadow skipped").
ORB       shadow trade khula (break tick, ML pass) -> MT5 MARKET order (SL range ki doosri side, TP 2R).
          Shadow band (SL / TP / 16:00 NY / manual) aur real abhi khula -> market pe band.
LOG       data/live/real_trades.csv (har band real trade: shadow vs real entry, slippage pips, real exit, $ P&L, commission)
          data/live/exec_status.json (dashboard: switches, account, khule positions / pending orders, last errors)
          data/live/exec_state.json (restart pe yaad: kaunsa ticket kis trade ka)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = {"zone": 1001, "orb": 2002}
MAX_LOTS = 5.0
DEVIATION = 20                      # points, market orders
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
REAL_COLS = ["strategy", "symbol", "direction", "key", "lots", "shadow_fill_time", "shadow_entry", "real_open_time",
             "real_entry", "entry_slip_pips", "sl", "risk", "shadow_r", "shadow_exit", "real_exit_time", "real_exit",
             "real_exit_reason", "real_r", "profit_usd", "commission_usd", "swap_usd", "net_usd", "ticket", "note"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pip(sym):
    return PIP.get(sym, 1e-2 if sym.endswith("JPY") else 1e-4)


# ======================================================================= broker
class MT5Broker:
    """Thin wrapper over the MetaTrader5 package (same terminal the feed uses)."""

    def __init__(self, mt5, log=print):
        self.mt5, self.log = mt5, log

    def is_demo(self) -> bool:
        a = self.mt5.account_info()
        return a is not None and a.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO

    def account(self) -> dict:
        a = self.mt5.account_info()
        if a is None:
            return {}
        return dict(login=a.login, server=a.server, demo=a.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO,
                    balance=round(a.balance, 2), equity=round(a.equity, 2), margin=round(getattr(a, "margin", 0.0), 2),
                    margin_free=round(a.margin_free, 2), margin_level=round(getattr(a, "margin_level", 0.0) or 0.0, 1),
                    leverage=getattr(a, "leverage", None), profit=round(getattr(a, "profit", 0.0), 2),
                    currency=a.currency, trade_allowed=bool(getattr(a, "trade_allowed", True)))

    def _sym(self, symbol):
        info = self.mt5.symbol_info(symbol)
        if info is None or not info.visible:
            self.mt5.symbol_select(symbol, True)
            info = self.mt5.symbol_info(symbol)
        return info

    def _filling(self, info):
        fm = int(getattr(info, "filling_mode", 0) or 0)
        if fm & 2:
            return self.mt5.ORDER_FILLING_IOC
        if fm & 1:
            return self.mt5.ORDER_FILLING_FOK
        return self.mt5.ORDER_FILLING_RETURN

    def _vol(self, info, lots):
        step = info.volume_step or 0.01
        v = min(float(lots), MAX_LOTS, info.volume_max or MAX_LOTS)
        v = np.floor(v / step + 1e-9) * step
        return round(v, 2) if v >= (info.volume_min or 0.01) - 1e-9 else 0.0

    def _send(self, req) -> tuple[bool, object, str]:
        """order_send; if the broker rejects the filling mode (10030), retry with the other modes."""
        modes = [req.get("type_filling")] if "type_filling" in req else [None]
        if "type_filling" in req:
            modes += [m for m in (self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_FOK)
                      if m != req["type_filling"]]
        err = ""
        for m in modes:
            if m is not None:
                req = dict(req, type_filling=m)
            r = self.mt5.order_send(req)
            if r is None:
                return False, None, f"order_send None: {self.mt5.last_error()}"
            ok = r.retcode in (self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL)
            if ok:
                return True, r, ""
            err = f"retcode {r.retcode} {r.comment}"
            if r.retcode != getattr(self.mt5, "TRADE_RETCODE_INVALID_FILL", 10030):
                break
        return False, None, err

    def place_limit(self, symbol, bull, price, sl, tp, lots, magic, comment) -> tuple[int | None, str]:
        info = self._sym(symbol)
        if info is None:
            return None, "symbol not found"
        vol = self._vol(info, lots)
        if vol <= 0:
            return None, "lot below broker minimum"
        d = info.digits
        req = dict(action=self.mt5.TRADE_ACTION_PENDING, symbol=symbol, volume=vol,
                   type=self.mt5.ORDER_TYPE_BUY_LIMIT if bull else self.mt5.ORDER_TYPE_SELL_LIMIT,
                   price=round(price, d), sl=round(sl, d), tp=round(tp, d) if tp else 0.0, magic=magic,
                   comment=comment[:31], type_time=self.mt5.ORDER_TIME_GTC, type_filling=self.mt5.ORDER_FILLING_RETURN)
        ok, r, err = self._send(req)
        return (int(r.order), "") if ok else (None, err)

    def market(self, symbol, bull, sl, tp, lots, magic, comment) -> tuple[int | None, float | None, str]:
        info = self._sym(symbol)
        if info is None:
            return None, None, "symbol not found"
        vol = self._vol(info, lots)
        if vol <= 0:
            return None, None, "lot below broker minimum"
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or not tick.ask:
            return None, None, "no price (market closed?)"
        d = info.digits
        req = dict(action=self.mt5.TRADE_ACTION_DEAL, symbol=symbol, volume=vol,
                   type=self.mt5.ORDER_TYPE_BUY if bull else self.mt5.ORDER_TYPE_SELL,
                   price=tick.ask if bull else tick.bid, sl=round(sl, d), tp=round(tp, d) if tp else 0.0,
                   deviation=DEVIATION, magic=magic, comment=comment[:31], type_time=self.mt5.ORDER_TIME_GTC,
                   type_filling=self._filling(info))
        ok, r, err = self._send(req)
        if not ok:
            return None, None, err
        pos = self.position_by_order(int(r.order), symbol, magic)
        return (pos.ticket if pos else int(r.order)), (pos.price_open if pos else float(r.price)), ""

    def cancel(self, order_ticket) -> str:
        ok, _, err = self._send(dict(action=self.mt5.TRADE_ACTION_REMOVE, order=int(order_ticket)))
        return "" if ok else err

    def modify(self, pos_ticket, symbol, sl, tp) -> str:
        info = self._sym(symbol)
        d = info.digits if info else 5
        ok, _, err = self._send(dict(action=self.mt5.TRADE_ACTION_SLTP, position=int(pos_ticket), symbol=symbol,
                                     sl=round(sl, d) if sl else 0.0, tp=round(tp, d) if tp else 0.0))
        return "" if ok else err

    def close(self, pos_ticket, magic) -> str:
        p = self.position(pos_ticket)
        if p is None:
            return ""                                   # already closed
        info = self._sym(p.symbol)
        tick = self.mt5.symbol_info_tick(p.symbol)
        if tick is None or not tick.bid:
            return "no price (market closed?)"
        buy = p.type == self.mt5.POSITION_TYPE_BUY
        req = dict(action=self.mt5.TRADE_ACTION_DEAL, symbol=p.symbol, volume=p.volume, position=int(pos_ticket),
                   type=self.mt5.ORDER_TYPE_SELL if buy else self.mt5.ORDER_TYPE_BUY,
                   price=tick.bid if buy else tick.ask, deviation=DEVIATION, magic=magic, comment="bot close",
                   type_time=self.mt5.ORDER_TIME_GTC, type_filling=self._filling(info))
        ok, _, err = self._send(req)
        return "" if ok else err

    def position(self, ticket):
        r = self.mt5.positions_get(ticket=int(ticket))
        return r[0] if r else None

    def position_by_order(self, order_ticket, symbol, magic):
        for p in self.mt5.positions_get(symbol=symbol) or []:
            if p.magic == magic and (p.identifier == order_ticket or p.ticket == order_ticket):
                return p
        return None

    def order(self, ticket):
        r = self.mt5.orders_get(ticket=int(ticket))
        return r[0] if r else None

    def positions(self, magic):
        return [p for p in (self.mt5.positions_get() or []) if p.magic == magic]

    def orders(self, magic):
        return [o for o in (self.mt5.orders_get() or []) if o.magic == magic]

    def closed_info(self, pos_ticket) -> dict | None:
        """Exit price / profit / commission / swap of a closed position (from deal history)."""
        deals = self.mt5.history_deals_get(position=int(pos_ticket))
        if not deals:
            frm = time.time() - 10 * 86400
            deals = [d for d in (self.mt5.history_deals_get(frm, time.time() + 86400) or []) if d.position_id == int(pos_ticket)]
        if not deals:
            return None
        outs = [d for d in deals if d.entry in (self.mt5.DEAL_ENTRY_OUT, self.mt5.DEAL_ENTRY_OUT_BY)]
        if not outs:
            return None
        last = outs[-1]
        reason = {getattr(self.mt5, "DEAL_REASON_SL", -1): "sl", getattr(self.mt5, "DEAL_REASON_TP", -2): "tp"}.get(last.reason, "close")
        return dict(exit_price=last.price, exit_ms=int(last.time_msc), reason=reason,
                    profit=round(sum(d.profit for d in deals), 2), commission=round(sum(d.commission for d in deals), 2),
                    swap=round(sum(d.swap for d in deals), 2), fee=round(sum(getattr(d, "fee", 0.0) for d in deals), 2))


# ======================================================================= shared state + logging
class Executor:
    """Owns the broker + both strategy books. run_live: ex.sync_zone(eng, prices); orb engine calls on_orb_open/close."""

    def __init__(self, broker, live: Path, log=print):
        self.b, self.live, self.log = broker, live, log
        self.state_file = live / "exec_state.json"
        self.status_file = live / "exec_status.json"
        self.trades_file = live / "real_trades.csv"
        self.zone: dict = {}            # key "SYM|zone|dir" -> rec
        self.orb: dict = {}             # key "SYM|fill_time" -> rec
        self.errors: list = []
        self.switch = {"zone": False, "orb": False}
        self._ctl_mtime = None
        self.demo_ok = False
        self._load()
        self.refresh_switch()

    # ---- switches (dashboard -> control.json "exec") ----
    def refresh_switch(self) -> bool:
        from src.live import control
        p = self.live / control.FILE
        try:
            mt = p.stat().st_mtime if p.exists() else None
        except OSError:
            return False
        if mt == self._ctl_mtime:
            return False
        self._ctl_mtime = mt
        new = control.load(self.live).get("exec") or {}
        new = {"zone": bool(new.get("zone")), "orb": bool(new.get("orb"))}
        changed = new != self.switch
        if changed:
            for k in ("zone", "orb"):
                if new[k] != self.switch[k]:
                    self.log(f"EXECUTION {k.upper()} -> {'ON' if new[k] else 'OFF'}")
                    control.log_action(self.live, kind=f"exec_{'on' if new[k] else 'off'}", symbol=k.upper())
            self.switch = new
        self.demo_ok = bool(self.b.is_demo())
        if any(self.switch.values()) and not self.demo_ok:
            self._err("SAFETY: account is not DEMO - no real orders")
        return changed

    def on(self, strat) -> bool:
        return self.switch.get(strat, False) and self.demo_ok

    def _err(self, msg):
        m = f"{pd.Timestamp.now(tz='UTC'):%m-%d %H:%M:%S} {msg}"
        if not self.errors or self.errors[-1][15:] != m[15:]:
            self.errors = (self.errors + [m])[-30:]
            self.log(f"EXEC: {msg}")

    def _save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(zone=self.zone, orb=self.orb, errors=self.errors), default=str))
        tmp.replace(self.state_file)

    def _load(self):
        try:
            st = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        except Exception:  # noqa: BLE001
            st = {}
        self.zone, self.orb, self.errors = st.get("zone", {}), st.get("orb", {}), st.get("errors", [])

    def _record(self, strat, rec, info: dict | None, note=""):
        """Write one closed real trade to real_trades.csv."""
        bull = rec["bull"]
        risk = float(rec.get("risk") or 0) or None
        rx = info.get("exit_price") if info else None
        re_ = rec.get("real_entry")
        real_r = ((rx - re_) if bull else (re_ - rx)) / risk if (rx is not None and re_ is not None and risk) else None
        net = (info["profit"] + info["commission"] + info["swap"] + info.get("fee", 0)) if info else None
        row = dict(strategy=strat, symbol=rec["symbol"], direction="bull" if bull else "bear", key=rec["key"],
                   lots=rec.get("lots"), shadow_fill_time=rec.get("shadow_fill_time"), shadow_entry=rec.get("shadow_entry"),
                   real_open_time=rec.get("real_open_time"), real_entry=re_, entry_slip_pips=rec.get("entry_slip_pips"),
                   sl=rec.get("sl"), risk=risk, shadow_r=rec.get("shadow_r"), shadow_exit=rec.get("shadow_exit"),
                   real_exit_time=pd.Timestamp(info["exit_ms"], unit="ms", tz="UTC").isoformat(timespec="seconds") if info else None,
                   real_exit=rx, real_exit_reason=info.get("reason") if info else None,
                   real_r=round(real_r, 3) if real_r is not None else None,
                   profit_usd=info.get("profit") if info else None, commission_usd=info.get("commission") if info else None,
                   swap_usd=info.get("swap") if info else None, net_usd=round(net, 2) if net is not None else None,
                   ticket=rec.get("position"), note=(rec.get("note") or "") + (f"; {note}" if note else ""))
        pd.DataFrame([row]).reindex(columns=REAL_COLS).to_csv(self.trades_file, mode="a",
                                                              header=not self.trades_file.exists(), index=False)

    def _slip(self, sym, bull, shadow_px, real_px):
        """+ = real fill WORSE than shadow (paid more on a buy / got less on a sell)."""
        if shadow_px is None or real_px is None:
            return None
        d = (real_px - shadow_px) if bull else (shadow_px - real_px)
        return round(d / _pip(sym), 2)

    # ================================================================ ZONE BOT
    @staticmethod
    def zkey(symbol, zone, direction):
        return f"{symbol}|{int(pd.Timestamp(zone).timestamp())}|{direction}"

    def sync_zone(self, eng) -> bool:
        """Every ~1 s: MT5 pending orders follow the shadow's armed orders; fills / closes / SL-TP follow shadow trades."""
        changed = False
        magic = MAGIC["zone"]
        on = self.on("zone")
        # 1) desired pending orders = armed orders of the current candle
        want = {}
        if on:
            for sym, orders in eng.armed.items():
                cur = eng._last_bars.get(sym, (None, None))[1]
                for o in orders:
                    if cur is None or o["bar_time"] == cur:
                        want[self.zkey(sym, o["zone"], o["direction"])] = (sym, o)
        # 2) place new
        for k, (sym, o) in want.items():
            rec = self.zone.get(k)
            if rec and rec.get("status") in ("pending", "open"):
                continue
            if rec and rec.get("status") == "failed" and _now_ms() - rec.get("at_ms", 0) < 300_000:
                continue                                   # failed recently -> don't spam
            bull = o["bull"]
            tp = o["entry"] + (1 if bull else -1) * o["risk"] if o["exit"] == "tp1" else None
            tk, err = self.b.place_limit(sym, bull, o["entry"], o["sl"], tp, o["lots"], magic, f"zone {k.split('|')[1]}")
            self.zone[k] = dict(key=k, symbol=sym, bull=bull, status="pending" if tk else "failed", order=tk,
                                entry=o["entry"], sl=o["sl"], tp=tp, risk=o["risk"], lots=o["lots"], exit=o["exit"],
                                at_ms=_now_ms(), error=err)
            if tk:
                self.log(f"EXEC ZONE: {sym} {'BUY' if bull else 'SELL'} LIMIT {o['entry']} SL {o['sl']} lots {o['lots']} -> order {tk}")
            else:
                self._err(f"ZONE {sym} limit order failed: {err}")
            changed = True
        # 3) pending orders: filled? no longer wanted?
        open_keys = {self.zkey(t["symbol"], t["zone"], t["direction"]): t for t in eng.open}
        for k, rec in list(self.zone.items()):
            if rec["status"] != "pending":
                continue
            od = self.b.order(rec["order"]) if rec.get("order") else None
            if od is None:                                       # gone: filled or cancelled
                pos = self.b.position_by_order(rec["order"], rec["symbol"], magic)
                if pos is not None:
                    rec.update(status="open", position=pos.ticket, real_entry=pos.price_open,
                               real_open_time=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), fill_seen_ms=_now_ms())
                    self.log(f"EXEC ZONE: {rec['symbol']} filled @ {pos.price_open} (position {pos.ticket})")
                else:
                    rec["status"] = "gone"
                changed = True
            elif k not in want:
                if k in open_keys and _now_ms() - rec.get("shadow_fill_ms", _now_ms()) < 60_000:
                    rec.setdefault("shadow_fill_ms", _now_ms())     # shadow just filled -> give the broker fill a minute
                    continue
                err = self.b.cancel(rec["order"])
                if err:
                    if self.b.order(rec["order"]) is not None:
                        self._err(f"ZONE {rec['symbol']} cancel failed: {err}")
                        continue
                rec["status"] = "cancelled"
                if k in open_keys:
                    rec["note"] = "shadow filled but the broker limit did not (missed)"
                    self._err(f"ZONE {rec['symbol']} real limit not filled while shadow filled - real trade missed")
                changed = True
        # 4) open real positions: follow the shadow trade
        for k, rec in list(self.zone.items()):
            if rec["status"] != "open":
                continue
            pos = self.b.position(rec["position"])
            t = open_keys.get(k)
            if t is not None and "shadow_entry" not in rec:
                rec.update(shadow_entry=t["entry"], shadow_fill_time=str(t.get("filled_at") or t.get("fill_time")),
                           entry_slip_pips=self._slip(rec["symbol"], rec["bull"], t["entry"], rec["real_entry"]))
            if pos is None:                                   # broker closed it (SL / TP) or we did
                self._finish_zone(k, rec, eng)
                changed = True
                continue
            if t is None:
                shadow_closed = next((c for c in reversed(eng.closed[-50:])
                                      if self.zkey(c["symbol"], c["zone"], c["direction"]) == k), None)
                if shadow_closed is not None or _now_ms() - rec.get("fill_seen_ms", 0) > 15_000:
                    why = "shadow trade closed" if shadow_closed is not None else "shadow did not take this fill"
                    err = self.b.close(rec["position"], magic)
                    if err:
                        self._err(f"ZONE {rec['symbol']} close failed: {err}")
                        continue
                    rec["note"] = why
                    self._finish_zone(k, rec, eng)
                    changed = True
                continue
            # SL / TP follow the shadow (trail moves SL, exit switch changes TP)
            want_sl = t.get("stop_now") or t["sl"]
            want_tp = t.get("tp") if t.get("exit") != "trail" else None
            pip = _pip(rec["symbol"])
            if abs((pos.sl or 0) - want_sl) > 0.3 * pip or abs((pos.tp or 0) - (want_tp or 0)) > 0.3 * pip:
                err = self.b.modify(rec["position"], rec["symbol"], want_sl, want_tp)
                if err:
                    self._err(f"ZONE {rec['symbol']} SL/TP modify failed: {err}")
                else:
                    changed = True
        # 5) cleanup old non-active records
        for k in [k for k, r in self.zone.items() if r["status"] in ("gone", "cancelled", "failed", "done")
                  and _now_ms() - r.get("at_ms", 0) > 86_400_000]:
            del self.zone[k]
        if changed:
            self._save()
        return changed

    def _finish_zone(self, k, rec, eng):
        info = self.b.closed_info(rec["position"])
        sc = next((c for c in reversed(eng.closed[-50:]) if self.zkey(c["symbol"], c["zone"], c["direction"]) == k), None)
        if sc is not None:
            rec.update(shadow_r=sc.get("r"), shadow_exit=sc.get("exit_reason"))
        self._record("zone", rec, info)
        rec["status"] = "done"
        self.log(f"EXEC ZONE: {rec['symbol']} real trade closed ({info.get('reason') if info else '?'}) net "
                 f"${(info['profit'] + info['commission'] + info['swap']) if info else 0:+.2f}")

    # ================================================================ ORB
    @staticmethod
    def okey(t):
        return f"{t['symbol']}|{int(pd.Timestamp(t['fill_time']).timestamp())}"

    def on_orb_open(self, tr: dict):
        if not self.on("orb"):
            return
        k = self.okey(tr)
        tk, px, err = self.b.market(tr["symbol"], tr["bull"], tr["sl"], tr["tp"], tr["lots"], MAGIC["orb"], f"orb {tr['range']}")
        if tk is None:
            self._err(f"ORB {tr['symbol']} market order failed: {err}")
            tr["real_error"] = err
            self.orb[k] = dict(key=k, symbol=tr["symbol"], bull=tr["bull"], status="failed", error=err, at_ms=_now_ms())
        else:
            slip = self._slip(tr["symbol"], tr["bull"], tr["entry"], px)
            tr.update(real_entry=px, real_slip_pips=slip, real_ticket=tk)
            self.orb[k] = dict(key=k, symbol=tr["symbol"], bull=tr["bull"], status="open", position=tk, real_entry=px,
                               shadow_entry=tr["entry"], shadow_fill_time=tr["fill_time"], entry_slip_pips=slip,
                               sl=tr["sl"], risk=tr["risk"], lots=tr["lots"], at_ms=_now_ms(),
                               real_open_time=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
            self.log(f"EXEC ORB: {tr['symbol']} {'BUY' if tr['bull'] else 'SELL'} market @ {px} (shadow {tr['entry']}, slip {slip} pips)")
        self._save()

    def on_orb_close(self, t: dict, r: float, why: str):
        k = self.okey(t)
        rec = self.orb.get(k)
        if not rec or rec.get("status") != "open":
            return
        if self.b.position(rec["position"]) is not None:          # SL / TP not hit at the broker yet -> close now
            err = self.b.close(rec["position"], MAGIC["orb"])
            if err:
                self._err(f"ORB {t['symbol']} close failed: {err}")
        rec.update(shadow_r=round(float(r), 3), shadow_exit=why)
        rec["status"] = "closing"
        self._save()

    def sync_orb(self) -> bool:
        """Every few seconds: real ORB positions that are closed -> record."""
        changed = False
        for k, rec in list(self.orb.items()):
            if rec.get("status") not in ("open", "closing"):
                continue
            if self.b.position(rec["position"]) is not None:
                continue
            if rec["status"] == "open":                              # broker SL/TP before the shadow saw it
                rec["note"] = "closed at the broker before the shadow"
            info = self.b.closed_info(rec["position"])
            if info is None and _now_ms() - rec.get("at_ms", 0) < 60_000:
                continue
            self._record("orb", rec, info)
            rec["status"] = "done"
            changed = True
        for k in [k for k, r in self.orb.items() if r.get("status") in ("done", "failed") and _now_ms() - r.get("at_ms", 0) > 86_400_000]:
            del self.orb[k]
        if changed:
            self._save()
        return changed

    def cancel_zone_pending(self):
        for rec in self.zone.values():
            if rec.get("status") == "pending":
                if not self.b.cancel(rec["order"]):
                    rec["status"] = "cancelled"
        self._save()

    # ================================================================ dashboard status
    def write_status(self):
        try:
            pos = []
            for strat in ("zone", "orb"):
                for p in self.b.positions(MAGIC[strat]):
                    pos.append(dict(strategy=strat, ticket=p.ticket, symbol=p.symbol, buy=p.type == 0, lots=p.volume,
                                    price_open=p.price_open, price_now=getattr(p, "price_current", None), sl=p.sl, tp=p.tp,
                                    profit=round(p.profit, 2), swap=round(getattr(p, "swap", 0.0), 2),
                                    time=int(getattr(p, "time", 0) or 0)))
            orders = [dict(strategy="zone", ticket=o.ticket, symbol=o.symbol, buy=getattr(o, "type", 2) in (0, 2, 4),
                           price=o.price_open, sl=o.sl, tp=o.tp, lots=o.volume_current, time=int(getattr(o, "time_setup", 0) or 0))
                      for o in self.b.orders(MAGIC["zone"])]
            links = {"zone": {k: {kk: r.get(kk) for kk in ("status", "order", "position", "real_entry", "entry_slip_pips", "error")}
                              for k, r in self.zone.items() if r.get("status") in ("pending", "open", "failed")},
                     "orb": {k: {kk: r.get(kk) for kk in ("status", "position", "real_entry", "entry_slip_pips", "error")}
                             for k, r in self.orb.items() if r.get("status") in ("open", "closing", "failed")}}
            st = dict(updated=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), switch=self.switch,
                      demo=self.demo_ok, account=self.b.account(), positions=pos, orders=orders, links=links,
                      errors=self.errors[-10:])
            tmp = self.status_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, default=str))
            tmp.replace(self.status_file)
        except Exception as e:  # noqa: BLE001
            self.log(f"exec status: {e}")
