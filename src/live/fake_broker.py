r"""FAKE BROKER for testing src/live/execution.py on replay data (no MT5). Same methods as MT5Broker.
Prices = last Dukascopy tick before the replay clock. process() fills limits / hits SL-TP at the current price."""
from __future__ import annotations

import itertools
from types import SimpleNamespace as NS

import pandas as pd

from src.backtest import tick_engine as te


class FakeBroker:
    def __init__(self, feed, demo=True):
        self.feed, self.demo = feed, demo
        self._orders, self.pos, self.deals = {}, {}, {}
        self._id = itertools.count(1000)
        self.log = []

    def _px(self, symbol):
        now = int(self.feed.clock.value // 1_000_000)
        tk = te.ticks(symbol, now - 3_600_000, now)
        return (float(tk[1][-1]), float(tk[2][-1])) if tk is not None and len(tk[0]) else (None, None)

    def is_demo(self):
        return self.demo

    def account(self):
        return dict(login=1, server="FAKE", demo=self.demo, balance=10000.0, equity=10000.0, margin_free=9000.0, currency="USD")

    def place_limit(self, symbol, bull, price, sl, tp, lots, magic, comment):
        t = next(self._id)
        self._orders[t] = NS(ticket=t, type=2 if bull else 3, symbol=symbol, bull=bull, price_open=price, sl=sl, tp=tp or 0.0, volume_current=lots,
                            magic=magic, comment=comment)
        self.log.append(("limit", symbol, price))
        return t, ""

    def _open(self, symbol, bull, px, sl, tp, lots, magic, ident):
        t = next(self._id)
        self.pos[t] = NS(ticket=t, identifier=ident, symbol=symbol, type=0 if bull else 1, volume=lots, price_open=px,
                         sl=sl, tp=tp or 0.0, profit=0.0, magic=magic)
        return t

    def market(self, symbol, bull, sl, tp, lots, magic, comment):
        ask, bid = self._px(symbol)
        if ask is None:
            return None, None, "no price"
        px = ask if bull else bid
        t = self._open(symbol, bull, px, sl, tp, lots, magic, None)
        self.pos[t].identifier = t
        self.log.append(("market", symbol, px))
        return t, px, ""

    def cancel(self, ticket):
        self._orders.pop(int(ticket), None)
        self.log.append(("cancel", ticket))
        return ""

    def modify(self, ticket, symbol, sl, tp):
        p = self.pos.get(int(ticket))
        if p:
            p.sl, p.tp = sl or 0.0, tp or 0.0
            self.log.append(("modify", symbol, sl, tp))
        return ""

    def _close_at(self, p, px, reason):
        mult = 100000 if p.symbol.endswith("USD") else 100000 / px
        prof = ((px - p.price_open) if p.type == 0 else (p.price_open - px)) * p.volume * mult
        self.deals[p.ticket] = dict(exit_price=px, exit_ms=int(self.feed.clock.value // 1_000_000), reason=reason,
                                    profit=round(prof, 2), commission=round(-5.0 * p.volume, 2), swap=0.0, fee=0.0)
        self.pos.pop(p.ticket, None)
        self.log.append(("close", p.symbol, px, reason))

    def close(self, ticket, magic):
        p = self.pos.get(int(ticket))
        if p is None:
            return ""
        ask, bid = self._px(p.symbol)
        self._close_at(p, bid if p.type == 0 else ask, "close")
        return ""

    def process(self):
        for t, o in list(self._orders.items()):
            ask, bid = self._px(o.symbol)
            if ask is None:
                continue
            if (o.bull and ask <= o.price_open) or (not o.bull and bid >= o.price_open):
                self._orders.pop(t)
                self._open(o.symbol, o.bull, o.price_open, o.sl, o.tp, o.volume_current, o.magic, t)
                self.log.append(("fill", o.symbol, o.price_open))
        for p in list(self.pos.values()):
            ask, bid = self._px(p.symbol)
            if ask is None:
                continue
            px = bid if p.type == 0 else ask
            if p.sl and ((p.type == 0 and px <= p.sl) or (p.type == 1 and px >= p.sl)):
                self._close_at(p, p.sl, "sl")
            elif p.tp and ((p.type == 0 and px >= p.tp) or (p.type == 1 and px <= p.tp)):
                self._close_at(p, p.tp, "tp")

    def position(self, ticket):
        return self.pos.get(int(ticket))

    def position_by_order(self, order_ticket, symbol, magic):
        return next((p for p in self.pos.values() if p.identifier == order_ticket and p.magic == magic), None)

    def order(self, ticket):
        return self._orders.get(int(ticket))

    def positions(self, magic):
        return [p for p in self.pos.values() if p.magic == magic]

    def orders(self, magic):
        return [o for o in self._orders.values() if o.magic == magic]

    def closed_info(self, ticket):
        return self.deals.get(int(ticket))
