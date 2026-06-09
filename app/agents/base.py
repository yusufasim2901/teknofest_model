"""
Abstract base class for all MAS agents.

Concrete agents inherit from :class:`BaseAgent` and implement
:meth:`on_event` to react to incoming :class:`AgentEvent` messages.
The base class provides lifecycle hooks and a convenience
:meth:`publish` method that delegates to the shared EventBus.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from app.models.events import AgentEvent

if TYPE_CHECKING:
    from app.main import EventBus

logger: logging.Logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract agent with lifecycle hooks and EventBus integration.

    Parameters:
        name:      Human-readable agent identifier (e.g. ``"perception_agent"``).
        event_bus: Shared :class:`EventBus` instance for publish/subscribe.
    """

    def __init__(self, name: str, event_bus: EventBus) -> None:
        self.name: str = name
        self._event_bus: EventBus = event_bus
        self._logger: logging.Logger = logging.getLogger(
            f"{__name__}.{self.name}",
        )

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self, subscribe_keys: list[str] | None = None) -> None:
        """Start the agent: subscribe to relevant routing keys.

        Args:
            subscribe_keys: List of topic routing patterns to listen on
                            (e.g. ``["perception.#", "network.qos_updated"]``).
        """
        self._logger.info("Agent '%s' starting…", self.name)
        if subscribe_keys:
            for key in subscribe_keys:
                await self._event_bus.subscribe(
                    routing_key=key,
                    callback=self._handle_delivery,
                )
                self._logger.info(
                    "Agent '%s' subscribed to '%s'.",
                    self.name,
                    key,
                )

    async def stop(self) -> None:
        """Graceful shutdown hook — override for cleanup logic."""
        self._logger.info("Agent '%s' stopping.", self.name)

    # ── Event handling ───────────────────────────────────────

    async def _handle_delivery(self, event: AgentEvent) -> None:
        """Internal wrapper: deserialize, log, and delegate to subclass."""
        self._logger.debug(
            "[%s] Received event '%s' from '%s'.",
            self.name,
            event.event_type,
            event.source_agent,
        )
        try:
            await self.on_event(event)
        except Exception:
            self._logger.exception(
                "Agent '%s' failed while handling event '%s' (id=%s).",
                self.name,
                event.event_type,
                event.event_id,
            )

    @abstractmethod
    async def on_event(self, event: AgentEvent) -> None:
        """Process an incoming event.

        Subclasses **must** implement this method.

        Args:
            event: The deserialized :class:`AgentEvent`.
        """
        ...

    # ── Publishing convenience ───────────────────────────────

    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        """Publish an event through the shared EventBus.

        Args:
            event_type:     Dot-namespaced routing key.
            payload:        JSON-serializable data dict.
            correlation_id: Optional correlation ID for tracing.
        """
        event = AgentEvent(
            source_agent=self.name,
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
        )
        await self._event_bus.publish(
            routing_key=event_type,
            event=event,
        )
        self._logger.debug(
            "[%s] Published event '%s' (id=%s).",
            self.name,
            event_type,
            event.event_id,
        )
