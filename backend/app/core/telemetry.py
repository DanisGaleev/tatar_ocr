from datetime import datetime, timezone
import logging
from typing import Any, Dict, List
import uuid
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class LiveTelemetryManager:
    """In-memory Pub/Sub broker managing active admin WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept WebSocket connection and add to active broadcaster pool."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Admin WebSocket connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection from broadcaster pool."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Admin WebSocket disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, event_data: Dict[str, Any]) -> None:
        """Broadcast event payload to all connected admin WebSockets."""
        payload = event_data.copy()
        if "event_id" not in payload:
            payload["event_id"] = f"evt_{uuid.uuid4().hex[:8]}"
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        dead_connections: List[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(payload)
            except Exception as exc:
                logger.warning(f"Error broadcasting to WebSocket: {exc}")
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)


telemetry_manager = LiveTelemetryManager()
