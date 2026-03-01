from __future__ import annotations


from fastapi.testclient import TestClient
import pytest


def test_websocket_connection(client: TestClient):
    with client.websocket_connect("/ws") as websocket:
        # Initial ping to verify connection is alive
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"


def test_websocket_pubsub_lifecycle(client: TestClient):
    with client.websocket_connect("/ws") as websocket:
        # Subscribe to run-2
        websocket.send_json({"type": "subscribe", "run_id": "run-2"})

        # The websocket will receive the "subscribed" message first
        subscribed_msg = websocket.receive_json()
        assert subscribed_msg["type"] == "subscribed"
        assert subscribed_msg["run_id"] == "run-2"

        # Unsubscribe
        websocket.send_json({"type": "unsubscribe", "run_id": "run-2"})
        unsubscribed_msg = websocket.receive_json()
        assert unsubscribed_msg["type"] == "unsubscribed"
        assert unsubscribed_msg["run_id"] == "run-2"


@pytest.mark.asyncio
async def test_manager_broadcast():
    from themis.server.app import ConnectionManager

    manager = ConnectionManager()

    class MockWebsocket:
        def __init__(self):
            self.messages = []

        async def accept(self):
            pass

        async def send_json(self, msg):
            self.messages.append(msg)

    ws1 = MockWebsocket()
    ws2 = MockWebsocket()

    await manager.connect(ws1)
    await manager.connect(ws2)

    manager.subscribe(ws1, "run-x")
    manager.subscribe(ws2, "run-y")

    # Broadcast to run-x
    event = {"type": "run_progress", "run_id": "run-x"}
    await manager.broadcast(event)

    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 0

    # Broadcast global
    await manager.broadcast({"type": "global"})
    assert len(ws1.messages) == 2
    assert len(ws2.messages) == 1


def test_websocket_invalid_message(client: TestClient):
    with client.websocket_connect("/ws") as websocket:
        # Send invalid JSON
        str_data = '{"type": "subscribe", missing_quotes}'

        # In Starlette/FastAPI TestClient, we have to send text if data is invalid JSON
        websocket.send_text(str_data)

        # Connection should respond with error or handle gracefully
        response = websocket.receive_json()
        assert response.get("type") == "error"
        assert "Invalid" in response.get("message", "")
