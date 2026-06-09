#!/usr/bin/env python3
"""
Proof-of-Concept: Non-blocking "Hello World" message exchange.

Demonstrates two dummy agents — **PerceptionAgent** and **DetectionAgent** —
communicating asynchronously through the EventBus over RabbitMQ.

PerceptionAgent publishes ``perception.hello`` every 2 seconds.
DetectionAgent subscribes to ``perception.#`` and logs each received message.

Usage::

    # Ensure RabbitMQ is running (e.g. via docker-compose up -d rabbitmq)
    python scripts/poc_hello_world.py

    # Or with custom RabbitMQ URL:
    RABBITMQ_HOST=localhost python scripts/poc_hello_world.py

Press Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from types import FrameType
from typing import Any

# ── Adjust sys.path so we can import from `app/` ────────────
# When running as `python scripts/poc_hello_world.py` from the project root,
# the parent directory (project root) needs to be on the path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.config import Settings  # noqa: E402
from app.main import EventBus  # noqa: E402
from app.models.events import AgentEvent  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger("poc")

# ──────────────────────────────────────────────────────────────────────
# Shutdown coordination
# ──────────────────────────────────────────────────────────────────────
_shutdown_event: asyncio.Event = asyncio.Event()


def _handle_signal(sig: int, _frame: FrameType | None = None) -> None:
    """Signal handler — sets the shutdown event so tasks exit cleanly."""
    logger.info("Received signal %s — initiating graceful shutdown…", sig)
    _shutdown_event.set()


# ──────────────────────────────────────────────────────────────────────
# Perception Agent (Publisher)
# ──────────────────────────────────────────────────────────────────────

async def perception_agent_loop(event_bus: EventBus) -> None:
    """Periodically publish ``perception.hello`` events.

    Runs until :data:`_shutdown_event` is set.

    Args:
        event_bus: Connected :class:`EventBus` instance.
    """
    counter: int = 0

    while not _shutdown_event.is_set():
        counter += 1

        event = AgentEvent(
            source_agent="perception_agent",
            event_type="perception.hello",
            payload={
                "message": f"Hello from PerceptionAgent! (seq={counter})",
                "frame_id": counter,
                "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

        await event_bus.publish(
            routing_key="perception.hello",
            event=event,
        )
        logger.info(
            "🔵 [PerceptionAgent] Published  → perception.hello  (seq=%d, id=%s)",
            counter,
            event.event_id,
        )

        # Non-blocking wait: exits early if shutdown is signalled
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pass  # Normal — timeout means "keep looping"


# ──────────────────────────────────────────────────────────────────────
# Detection Agent (Subscriber)
# ──────────────────────────────────────────────────────────────────────

_received_count: int = 0


async def detection_agent_callback(event: AgentEvent) -> None:
    """Callback invoked for each event matching ``perception.#``.

    Args:
        event: Deserialized :class:`AgentEvent` from the bus.
    """
    global _received_count  # noqa: PLW0603
    _received_count += 1

    logger.info(
        "🟢 [DetectionAgent] Received   ← %s  |  payload=%s  (total_rx=%d)",
        event.event_type,
        event.payload.get("message", ""),
        _received_count,
    )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Entrypoint: wire up the EventBus, start both agents, await shutdown."""
    # Build settings — honours RABBITMQ_HOST / RABBITMQ_PORT env overrides
    settings = Settings(
        rabbitmq_host=os.getenv("RABBITMQ_HOST", "localhost"),
        rabbitmq_default_user=os.getenv("RABBITMQ_DEFAULT_USER", "guest"),
        rabbitmq_default_pass=os.getenv("RABBITMQ_DEFAULT_PASS", "guest"),
    )

    event_bus = EventBus(settings)

    try:
        # ── Connect ──────────────────────────────────────────
        logger.info("━" * 60)
        logger.info("MAS PoC — Hello World (Non-Blocking Agent Exchange)")
        logger.info("━" * 60)

        await event_bus.connect()

        # ── Subscribe the DetectionAgent ─────────────────────
        await event_bus.subscribe(
            routing_key="perception.#",
            callback=detection_agent_callback,
        )
        logger.info("🟢 [DetectionAgent] Subscribed to 'perception.#'.")

        # ── Run PerceptionAgent concurrently ─────────────────
        logger.info("🔵 [PerceptionAgent] Starting publish loop (every 2s)…")
        logger.info("Press Ctrl+C to stop.\n")

        await perception_agent_loop(event_bus)

    except ConnectionError as exc:
        logger.error("Could not connect to RabbitMQ: %s", exc)
        logger.error(
            "Hint: Ensure RabbitMQ is running.  "
            "Try `docker-compose up -d rabbitmq` first.",
        )
        sys.exit(1)

    finally:
        # ── Teardown ─────────────────────────────────────────
        logger.info("\nShutting down…")
        await event_bus.disconnect()
        logger.info("✅ PoC finished.  Messages sent: ?, received: %d.", _received_count)


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    asyncio.run(main())
