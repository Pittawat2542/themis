"""WebSocket connection manager for Themis server."""

from __future__ import annotations

from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        for clients in self.subscriptions.values():
            clients.discard(websocket)

    def subscribe(self, websocket: WebSocket, run_id: str):
        if run_id not in self.subscriptions:
            self.subscriptions[run_id] = set()
        self.subscriptions[run_id].add(websocket)

    def unsubscribe(self, websocket: WebSocket, run_id: str):
        if run_id in self.subscriptions:
            self.subscriptions[run_id].discard(websocket)

    async def broadcast(self, message: dict):
        run_id = message.get("run_id")
        if run_id and run_id in self.subscriptions:
            # Targeted broadcast
            for connection in list(self.subscriptions[run_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(connection)
        elif not run_id:
            # Global broadcast
            for connection in list(self.active_connections):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(connection)


# Global manager instance
manager = ConnectionManager()
_manager = manager

__all__ = ["ConnectionManager", "manager", "_manager"]
