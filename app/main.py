"""
MAS — FastAPI Application & Asynchronous EventBus.

This module is the backbone of the Multi-Agent System.  It provides:

1.  **EventBus** — An async wrapper around ``aio-pika`` that manages a
    RabbitMQ *topic exchange*, exposes ``publish`` / ``subscribe`` methods,
    and handles connection recovery with exponential back-off.

2.  **FastAPI lifespan** — Initialises the EventBus on startup and tears
    it down gracefully on shutdown.

3.  **REST endpoints** — ``/health`` for liveness probes and
    ``/events/publish`` for manual event injection during development.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Awaitable

import aio_pika
from aio_pika import (
    DeliveryMode,
    ExchangeType,
    Message,
)
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.models.events import AgentEvent, PublishRequest
from app.routers.open_gateway import router as gateway_router

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger(__name__)

# Type alias for subscriber callbacks
EventCallback = Callable[[AgentEvent], Awaitable[None]]

# ──────────────────────────────────────────────────────────────────────
# EventBus
# ──────────────────────────────────────────────────────────────────────

# Default exchange name used across the MAS pipeline
_EXCHANGE_NAME: str = "mas.events"

# Connection retry parameters
_MAX_RETRIES: int = 5
_INITIAL_BACKOFF_S: float = 1.0
_BACKOFF_MULTIPLIER: float = 2.0


class EventBus:
    """Asynchronous message bus backed by a RabbitMQ *topic* exchange.

    The ``EventBus`` manages its own connection and channel lifecycle.
    It is designed to be used as a singleton — instantiated once in the
    FastAPI lifespan and shared across all agents.

    Usage::

        bus = EventBus(settings)
        await bus.connect()
        await bus.subscribe("perception.#", my_callback)
        await bus.publish("perception.frame_analyzed", event)
        ...
        await bus.disconnect()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._queues: list[AbstractQueue] = []

    # ── Connection lifecycle ─────────────────────────────────

    async def connect(self) -> None:
        """Open a robust AMQP connection and declare the topic exchange.

        Implements exponential back-off for transient failures
        (e.g. RabbitMQ not yet ready during container orchestration).

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        backoff: float = _INITIAL_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "EventBus — connecting to RabbitMQ (attempt %d/%d)…",
                    attempt,
                    _MAX_RETRIES,
                )
                self._connection = await aio_pika.connect_robust(
                    url=self._settings.rabbitmq_url,
                    timeout=10.0,
                )
                self._channel = await self._connection.channel()
                await self._channel.set_qos(prefetch_count=10)

                # Declare the durable topic exchange
                self._exchange = await self._channel.declare_exchange(
                    name=_EXCHANGE_NAME,
                    type=ExchangeType.TOPIC,
                    durable=True,
                )
                logger.info(
                    "EventBus — connected.  Exchange '%s' ready.",
                    _EXCHANGE_NAME,
                )
                return

            except (ConnectionError, OSError, aio_pika.exceptions.AMQPError) as exc:
                logger.warning(
                    "EventBus — connection attempt %d failed: %s",
                    attempt,
                    exc,
                )
                if attempt == _MAX_RETRIES:
                    msg = (
                        f"EventBus — failed to connect after "
                        f"{_MAX_RETRIES} attempts."
                    )
                    raise ConnectionError(msg) from exc
                await asyncio.sleep(backoff)
                backoff *= _BACKOFF_MULTIPLIER

    async def disconnect(self) -> None:
        """Gracefully close the channel and connection."""
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
            logger.info("EventBus — channel closed.")
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("EventBus — connection closed.")

    # ── Publish ──────────────────────────────────────────────

    async def publish(
        self,
        routing_key: str,
        event: AgentEvent,
    ) -> None:
        """Serialize and publish an event to the topic exchange.

        Args:
            routing_key: Dot-namespaced topic key (e.g. ``"perception.frame_analyzed"``).
            event:       The :class:`AgentEvent` to publish.

        Raises:
            RuntimeError: If the EventBus is not connected.
            ValueError:   If the event cannot be serialized.
        """
        if self._exchange is None:
            raise RuntimeError(
                "EventBus is not connected.  Call `connect()` first.",
            )

        try:
            body: bytes = event.model_dump_json().encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to serialize event {event.event_id}: {exc}",
            ) from exc

        message = Message(
            body=body,
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=str(event.event_id),
            correlation_id=(
                str(event.correlation_id) if event.correlation_id else None
            ),
            timestamp=event.timestamp,
        )

        await self._exchange.publish(
            message=message,
            routing_key=routing_key,
        )
        logger.debug(
            "EventBus — published '%s' (id=%s).",
            routing_key,
            event.event_id,
        )

    # ── Subscribe ────────────────────────────────────────────

    async def subscribe(
        self,
        routing_key: str,
        callback: EventCallback,
    ) -> AbstractQueue:
        """Bind an exclusive queue to the exchange and start consuming.

        Args:
            routing_key: Topic pattern (e.g. ``"perception.#"``).
            callback:    Async callable receiving a deserialized :class:`AgentEvent`.

        Returns:
            The declared queue (useful for testing / introspection).

        Raises:
            RuntimeError: If the EventBus is not connected.
        """
        if self._channel is None or self._exchange is None:
            raise RuntimeError(
                "EventBus is not connected.  Call `connect()` first.",
            )

        queue: AbstractQueue = await self._channel.declare_queue(
            name="",          # Let RabbitMQ generate a unique name
            exclusive=True,   # Auto-delete when the consumer disconnects
        )
        await queue.bind(exchange=self._exchange, routing_key=routing_key)

        async def _on_message(message: AbstractIncomingMessage) -> None:
            """Deserialize, dispatch, and acknowledge."""
            async with message.process():
                try:
                    raw: dict[str, Any] = json.loads(message.body.decode("utf-8"))
                    event = AgentEvent.model_validate(raw)
                    await callback(event)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.error(
                        "EventBus — failed to deserialize message: %s",
                        exc,
                    )

        await queue.consume(_on_message)
        self._queues.append(queue)

        logger.info(
            "EventBus — subscribed to '%s' → queue '%s'.",
            routing_key,
            queue.name,
        )
        return queue

    # ── Utilities ────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the AMQP connection is alive."""
        return (
            self._connection is not None
            and not self._connection.is_closed
        )


# ──────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────

# Module-level reference so endpoints can access the bus
_event_bus: EventBus | None = None
_redis: Redis | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: boot services on startup, tear down on shutdown."""
    global _event_bus, _redis  # noqa: PLW0603

    settings: Settings = get_settings()

    # ── Startup ──────────────────────────────────────────────
    logger.info("MAS API — starting up…")

    # EventBus
    _event_bus = EventBus(settings)
    await _event_bus.connect()

    # Redis
    _redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
    await _redis.ping()
    logger.info("Redis — connected.")

    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("MAS API — shutting down…")
    await _event_bus.disconnect()
    await _redis.aclose()


app = FastAPI(
    title="MAS — Multi-Agent System API",
    description=(
        "Event-driven backbone for a 5G-enabled smart road safety platform. "
        "Publish and consume agent events via RabbitMQ."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

# ── Mount routers ────────────────────────────────────────────────────
app.include_router(gateway_router)


# ── Endpoints ────────────────────────────────────────────────────────


@app.get(
    "/health",
    summary="Liveness probe",
    tags=["Infrastructure"],
)
async def health_check() -> dict[str, str]:
    """Check connectivity to RabbitMQ and Redis.

    Returns a JSON object with the status of each dependency.
    """
    rmq_status: str = "disconnected"
    redis_status: str = "disconnected"

    if _event_bus is not None and _event_bus.is_connected:
        rmq_status = "connected"

    if _redis is not None:
        try:
            await _redis.ping()
            redis_status = "connected"
        except Exception:  # noqa: BLE001
            redis_status = "error"

    overall: str = (
        "healthy"
        if rmq_status == "connected" and redis_status == "connected"
        else "degraded"
    )

    return {
        "status": overall,
        "rabbitmq": rmq_status,
        "redis": redis_status,
    }


@app.post(
    "/events/publish",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Manually inject an event (debug)",
    tags=["Events"],
)
async def publish_event(body: PublishRequest) -> dict[str, str]:
    """Publish an event to the EventBus for testing / debugging.

    This endpoint wraps the payload in an :class:`AgentEvent` envelope
    and publishes it to the topic exchange.
    """
    if _event_bus is None or not _event_bus.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EventBus is not connected.",
        )

    event = AgentEvent(
        source_agent=body.source_agent,
        event_type=body.routing_key,
        payload=body.payload,
    )
    await _event_bus.publish(routing_key=body.routing_key, event=event)

    return {
        "status": "accepted",
        "event_id": str(event.event_id),
    }


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    """WebSocket endpoint to stream real-time violation alerts to the frontend.

    1. Accepts the WebSocket connection.
    2. Subscribes to the `decision.violation_alert` topic on the EventBus.
    3. Forwards each received event as a JSON text frame to the client.
    4. Cleans up the subscription when the client disconnects.
    """
    await websocket.accept()

    if _event_bus is None or not _event_bus.is_connected:
        logger.warning("WebSocket connected but EventBus is unavailable.")
        await websocket.close(code=1011, reason="EventBus not connected.")
        return

    # Create an asyncio queue to act as a bridge between the EventBus callback
    # and the WebSocket send loop. This prevents blocking the EventBus consumer.
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)

    async def _on_alert(event: AgentEvent) -> None:
        try:
            # We forward the raw JSON payload to the client
            # (which is a ViolationAlert schema)
            payload_json: str = json.dumps(event.payload)
            queue.put_nowait(payload_json)
        except asyncio.QueueFull:
            logger.warning("WebSocket queue full — dropping alert %s.", event.event_id)
        except Exception as exc:
            logger.error("Error formatting alert for WebSocket: %s", exc)

    # Subscribe to the topic
    # We use a unique exclusive queue for each connected client
    rmq_queue = await _event_bus.subscribe(
        routing_key="decision.violation_alert",
        callback=_on_alert,
    )
    
    logger.info("WebSocket client connected and subscribed to alerts.")

    try:
        # Loop indefinitely, taking items from the queue and sending them
        while True:
            # We use wait_for to periodically check if the client disconnected
            # (though receive_text usually handles disconnect detection)
            try:
                alert_json = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_text(alert_json)
                queue.task_done()
            except asyncio.TimeoutError:
                pass # Just wake up and loop again
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally.")
    except Exception as exc:
        logger.warning("WebSocket connection dropped: %s", exc)
    finally:
        # Clean up the RabbitMQ subscription
        if rmq_queue is not None:
            await rmq_queue.delete()
        logger.info("WebSocket subscription cleaned up.")
