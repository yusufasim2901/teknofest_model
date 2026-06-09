"""
MAS Load Testing Suite — End-to-End Pipeline Latency Validation.

Simulates heavy concurrent video frame processing via RabbitMQ to ensure
the EventBus and message broker can sustain the required <100 ms per-frame
latency budget under load.

Requirements:
    - RabbitMQ must be running locally (e.g. ``docker-compose up -d rabbitmq``).

Usage:
    pytest tests/test_load_pipeline.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.main import EventBus
from app.models.events import AgentEvent

# ──────────────────────────────────────────────────────────────────────
# Test Config & Markers
# ──────────────────────────────────────────────────────────────────────

# Treat all tests in this file as async
pytestmark = pytest.mark.asyncio

# Target latency thresholds (milliseconds)
_LATENCY_P50_MAX_MS: float = 50.0
_LATENCY_P99_MAX_MS: float = 100.0


# ──────────────────────────────────────────────────────────────────────
# Load Tests
# ──────────────────────────────────────────────────────────────────────

async def test_50_concurrent_frames_under_100ms(event_bus: EventBus) -> None:
    """Validate end-to-end latency for 50 concurrent events.

    1. Subscribes a consumer to ``loadtest.frame``.
    2. Publishes 50 events concurrently using ``asyncio.gather``.
    3. The consumer records the wall-clock time difference between when
       the event was constructed and when it was received.
    4. Asserts p50 < 50ms and p99 < 100ms.
    """
    total_events: int = 50
    routing_key: str = "loadtest.frame"
    
    # Shared state for the consumer callback
    received_latencies_ms: list[float] = []
    completion_event = asyncio.Event()

    # ── Consumer callback ────────────────────────────────
    async def on_event(event: AgentEvent) -> None:
        """Calculate E2E latency from embedded sent timestamp."""
        receive_time: float = time.perf_counter()
        
        # Extract the sent timestamp injected by the publisher
        sent_time: float = event.payload.get("sent_timestamp", 0.0)
        
        latency_ms: float = (receive_time - sent_time) * 1000
        received_latencies_ms.append(latency_ms)

        # Signal completion when all expected events arrive
        if len(received_latencies_ms) == total_events:
            completion_event.set()

    # ── 1. Subscribe ─────────────────────────────────────
    queue = await event_bus.subscribe(routing_key, on_event)

    # Allow a brief moment for the queue binding to propagate in RabbitMQ
    await asyncio.sleep(0.1)

    # ── 2. Publish concurrently ──────────────────────────
    # We create the events just before publishing to accurately measure
    # the time spent serialising, transmitting, routing, and consuming.
    async def publish_one(seq: int) -> None:
        payload = {
            "seq": seq,
            "sent_timestamp": time.perf_counter(),
            "dummy_data": "x" * 1024,  # Simulate 1KB payload per frame
        }
        event = AgentEvent(
            source_agent="load_tester",
            event_type=routing_key,
            payload=payload,
        )
        await event_bus.publish(routing_key, event)

    # Fire all 50 publishes at exactly the same time
    await asyncio.gather(*(publish_one(i) for i in range(total_events)))

    # ── 3. Wait for consumption ──────────────────────────
    try:
        # Give the broker a max of 5 seconds to process 50 messages
        await asyncio.wait_for(completion_event.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail(
            f"Timeout: received {len(received_latencies_ms)}/{total_events} events."
        )

    # ── 4. Analyze & Assert ──────────────────────────────
    latencies = sorted(received_latencies_ms)
    min_ms = latencies[0]
    max_ms = latencies[-1]
    
    # Calculate percentiles manually (no numpy required for tests)
    def percentile(p: float) -> float:
        k = (len(latencies) - 1) * p
        f = int(k)
        c = int(k) + 1 if k > int(k) else int(k)
        if f == c:
            return latencies[f]
        return latencies[f] * (c - k) + latencies[c] * (k - f)

    p50_ms = percentile(0.50)
    p95_ms = percentile(0.95)
    p99_ms = percentile(0.99)

    # Print summary table for pytest output (requires -s or failing test)
    print("\n━━━ Latency Summary (50 events) ━━━━━")
    print(f"Min:  {min_ms:6.2f} ms")
    print(f"p50:  {p50_ms:6.2f} ms  (target: <{_LATENCY_P50_MAX_MS} ms)")
    print(f"p95:  {p95_ms:6.2f} ms")
    print(f"p99:  {p99_ms:6.2f} ms  (target: <{_LATENCY_P99_MAX_MS} ms)")
    print(f"Max:  {max_ms:6.2f} ms")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # The actual assertions
    assert p50_ms < _LATENCY_P50_MAX_MS, f"Median latency {p50_ms:.1f}ms exceeds target."
    assert p99_ms < _LATENCY_P99_MAX_MS, f"p99 latency {p99_ms:.1f}ms exceeds target."
    
    # We also assert that max is within a reasonable bound (200ms) to ensure
    # there are no catastrophic outliers caused by GC pauses or scheduling jitter.
    assert max_ms < 200.0, f"Max outlier {max_ms:.1f}ms is unacceptable."

    # Cleanup the queue (exclusive queues auto-delete, but manual cleanup is cleaner)
    await queue.delete()


async def test_concurrent_publish_does_not_block(event_bus: EventBus) -> None:
    """Verify that publishing is decoupled from consumption speed.

    If a consumer is slow, the publisher should not be blocked from
    enqueuing more messages onto the bus.
    """
    total_events: int = 20
    routing_key: str = "loadtest.nonblocking"
    
    # ── Slow consumer ────────────────────────────────────
    async def slow_on_event(_: AgentEvent) -> None:
        # Simulate heavy processing (e.g. ML inference) taking 50ms
        await asyncio.sleep(0.05)
        
    queue = await event_bus.subscribe(routing_key, slow_on_event)
    await asyncio.sleep(0.1)

    # ── Measure publish time ─────────────────────────────
    # We publish 20 events. If publish blocked on the consumer,
    # this would take 20 * 50ms = 1000ms.
    t0 = time.perf_counter()
    
    async def publish_one() -> None:
        event = AgentEvent(
            source_agent="load_tester",
            event_type=routing_key,
            payload={},
        )
        await event_bus.publish(routing_key, event)
        
    await asyncio.gather(*(publish_one() for _ in range(total_events)))
    
    publish_duration_ms = (time.perf_counter() - t0) * 1000
    
    # Assert publish completes in a fraction of the consumer time
    # (typically < 20ms for 20 events)
    assert publish_duration_ms < 100.0, (
        f"Publishing {total_events} events took {publish_duration_ms:.1f}ms, "
        "indicating it might be blocking on the slow consumer."
    )

    await queue.delete()
