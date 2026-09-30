"""
Event bus: agents ki live activity dashboard tak pahunchane ka system.

Kaise kaam karta hai:
  - Koi bhi code (jaise zone scan) `bus.emit(...)` bolta hai: "ZONE HUNTER ne ye kiya".
  - Event bus us message ko har connected dashboard (WebSocket) ko turant bhej deta hai.
  - Aakhri 200 events yaad rakhta hai, taaki naya dashboard khule toh purani activity bhi dikhe.

emit() thread-safe hai: background thread mein chal raha kaam bhi events bhej sakta hai.
"""

import asyncio
import time
from collections import deque

from fastapi import WebSocket


class EventBus:
    def __init__(self, history_size: int = 200):
        self.clients: set[WebSocket] = set()
        self.history: deque = deque(maxlen=history_size)
        self.loop: asyncio.AbstractEventLoop | None = None   # server start hone pe set hota hai

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        for event in list(self.history):          # naye dashboard ko purani activity bhi bhejo
            await ws.send_json(event)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    def emit(self, agent: str, message: str, status: str = "active",
             progress: float | None = None, kind: str = "log", data: dict | None = None) -> None:
        """
        agent    : "data" | "structure" | "zone" | "ml" | "risk" | "execution" | "system"
        message  : kya ho raha hai (dashboard pe English mein dikhega)
        status   : "active" | "idle" | "done" | "error"
        progress : 0-100 (progress bar ke liye), ya None
        kind     : "log" (normal activity) | "result" (test ka final result)
        data     : extra info (jaise result ke numbers)
        """
        event = {
            "ts": time.time(), "agent": agent, "message": message, "status": status,
            "progress": progress, "kind": kind, "data": data or {},
        }
        self.history.append(event)
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                self.clients.discard(ws)       # dashboard band ho gaya, hata do


bus = EventBus()
