"""
Event schemas for inter-agent communication.

Every message flowing through the EventBus is wrapped in an
:class:`AgentEvent` envelope that carries metadata for tracing,
correlation, and replay.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, UUID4


class AgentEvent(BaseModel):
    """Canonical envelope for all events exchanged between agents.

    Attributes:
        event_id:       Unique identifier for this event instance.
        source_agent:   Name of the agent that produced the event.
        event_type:     Dot-namespaced routing key (e.g. ``perception.frame_analyzed``).
        timestamp:      UTC creation timestamp.
        payload:        Arbitrary JSON-serializable data.
        correlation_id: Optional ID linking related events across the pipeline.
    """

    event_id: UUID4 = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    source_agent: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Name of the producing agent.",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        max_length=256,
        pattern=r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)*$",
        description="Dot-namespaced routing key.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC creation timestamp.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data.",
    )
    correlation_id: UUID4 | None = Field(
        default=None,
        description="Optional correlation ID for distributed tracing.",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra: dict[str, Any] = {
            "example": {
                "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "source_agent": "perception_agent",
                "event_type": "perception.frame_analyzed",
                "timestamp": "2026-06-09T12:00:00Z",
                "payload": {
                    "frame_id": 42,
                    "objects_detected": 3,
                },
                "correlation_id": "f0e1d2c3-b4a5-6789-0abc-def123456789",
            },
        }


class PublishRequest(BaseModel):
    """Schema for the manual ``POST /events/publish`` endpoint."""

    routing_key: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Topic routing key.",
    )
    source_agent: str = Field(
        default="api_manual",
        description="Logical name of the source.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON payload.",
    )
