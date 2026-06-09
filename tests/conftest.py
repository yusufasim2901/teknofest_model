"""
Shared pytest fixtures for the MAS test suite.

Provides session-scoped async fixtures for the EventBus and RabbitMQ
connection.  Tests that need a live RabbitMQ instance should be run
with the Docker stack up (``docker-compose up -d rabbitmq``).

Configuration is read from environment variables or ``.env``, with
defaults pointing to ``localhost`` for local development.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.config import Settings
from app.main import EventBus


# ──────────────────────────────────────────────────────────────────────
# Event Loop — Session Scope
# ──────────────────────────────────────────────────────────────────────
# pytest-asyncio needs a custom event loop fixture when using
# session-scoped async fixtures.  Without this, each test function
# would get its own loop, breaking shared connections.


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ──────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Build a Settings instance pointing to the test RabbitMQ.

    Defaults to ``localhost`` so tests run against a local Docker stack.
    Override via environment variables if needed.
    """
    return Settings(
        rabbitmq_host=os.getenv("TEST_RABBITMQ_HOST", "localhost"),
        rabbitmq_port=int(os.getenv("TEST_RABBITMQ_PORT", "5672")),
        rabbitmq_default_user=os.getenv("TEST_RABBITMQ_USER", "guest"),
        rabbitmq_default_pass=os.getenv("TEST_RABBITMQ_PASS", "guest"),
    )


# ──────────────────────────────────────────────────────────────────────
# EventBus
# ──────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="session")
async def event_bus(settings: Settings) -> AsyncGenerator[EventBus, None]:
    """Provide a connected EventBus for the entire test session.

    Connects on first use, disconnects after all tests complete.
    Tests share this connection to avoid per-test connection overhead
    (RabbitMQ connection setup takes ~50–100ms).
    """
    bus = EventBus(settings)
    await bus.connect()
    yield bus
    await bus.disconnect()
