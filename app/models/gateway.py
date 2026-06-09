"""
Pydantic schemas for GSMA Open Gateway API payloads.

Models mirror the CAMARA ``quality-on-demand`` v0.10 and
``number-verification`` v1.0 specifications, adapted for both
the mock service and the live gateway client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, UUID4, field_validator


# ──────────────────────────────────────────────────────────────────────
# QoS Profile Enum
# ──────────────────────────────────────────────────────────────────────

class QoSProfile(StrEnum):
    """CAMARA QoS profile identifiers.

    Each profile maps to a different latency / throughput guarantee
    on the 5G network slice.
    """

    QOS_E = "QOS_E"   # Ultra-low latency (≤ 10 ms)
    QOS_S = "QOS_S"   # Low latency     (≤ 50 ms)
    QOS_M = "QOS_M"   # Medium latency  (≤ 100 ms)
    QOS_L = "QOS_L"   # Standard        (best-effort)


class QoDSessionStatus(StrEnum):
    """Lifecycle status of a QoD session."""

    REQUESTED = "REQUESTED"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DELETED = "DELETED"


# ──────────────────────────────────────────────────────────────────────
# QoD — Request / Response
# ──────────────────────────────────────────────────────────────────────

class DeviceIdentifier(BaseModel):
    """Device identification by IPv4 address."""

    ipv4_address: str = Field(
        ...,
        description="IPv4 address of the device (UE).",
        examples=["10.0.0.1"],
    )


class ApplicationServer(BaseModel):
    """Application server identification."""

    ipv4_address: str = Field(
        ...,
        description="IPv4 address of the application server.",
        examples=["10.0.0.2"],
    )


class QoDSessionRequest(BaseModel):
    """Request body for ``POST /gateway/qod/sessions``.

    Follows CAMARA ``quality-on-demand`` v0.10 ``CreateSession`` schema.
    """

    device: DeviceIdentifier = Field(
        ...,
        description="Target device (UE) identifier.",
    )
    application_server: ApplicationServer = Field(
        ...,
        description="Application server to create the QoS flow towards.",
    )
    qos_profile: QoSProfile = Field(
        default=QoSProfile.QOS_E,
        description="Requested QoS profile.",
    )
    duration: int = Field(
        default=300,
        ge=1,
        le=86400,
        description="Session duration in seconds.",
    )
    webhook_url: str | None = Field(
        default=None,
        description="Optional callback URL for session status notifications.",
    )


class QoDSessionResponse(BaseModel):
    """Response body for QoD session operations.

    Returned by both ``POST`` (create) and ``GET`` (retrieve).
    """

    session_id: UUID4 = Field(
        default_factory=uuid4,
        description="Unique session identifier.",
    )
    status: QoDSessionStatus = Field(
        default=QoDSessionStatus.AVAILABLE,
        description="Current session status.",
    )
    qos_profile: QoSProfile = Field(
        description="Granted QoS profile.",
    )
    device: DeviceIdentifier = Field(
        description="Device associated with this session.",
    )
    application_server: ApplicationServer = Field(
        description="Application server endpoint.",
    )
    duration: int = Field(
        description="Session duration in seconds.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when the session was created.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the session expires.",
    )


# ──────────────────────────────────────────────────────────────────────
# Number Verification — Request / Response
# ──────────────────────────────────────────────────────────────────────

class NumberVerifyRequest(BaseModel):
    """Request body for ``POST /gateway/number-verification/verify``.

    Follows CAMARA ``number-verification`` v1.0 schema.
    """

    phone_number: str = Field(
        ...,
        min_length=8,
        max_length=16,
        description="Phone number in E.164 format (e.g. +905551234568).",
        examples=["+905551234568"],
    )
    hashed_token: str = Field(
        ...,
        min_length=1,
        description="OAuth2 access token hash for device authentication.",
    )

    @field_validator("phone_number")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        """Ensure the phone number starts with '+' and contains only digits."""
        if not v.startswith("+"):
            msg = "Phone number must start with '+' (E.164 format)."
            raise ValueError(msg)
        digits = v[1:]
        if not digits.isdigit():
            msg = "Phone number must contain only digits after '+'."
            raise ValueError(msg)
        return v


class NumberVerifyResponse(BaseModel):
    """Response body for number verification."""

    device_phone_number_verified: bool = Field(
        description="Whether the phone number matches the device.",
    )
    server_correlation_id: UUID4 = Field(
        default_factory=uuid4,
        description="Server-side correlation ID for audit trails.",
    )


# ──────────────────────────────────────────────────────────────────────
# CAMARA Error Response
# ──────────────────────────────────────────────────────────────────────

class GatewayErrorResponse(BaseModel):
    """Standard CAMARA error response envelope."""

    status: int = Field(description="HTTP status code.")
    code: str = Field(
        description="CAMARA error code (e.g. INVALID_ARGUMENT).",
    )
    message: str = Field(description="Human-readable error description.")


# ──────────────────────────────────────────────────────────────────────
# Decision Agent — Violation Payload
# ──────────────────────────────────────────────────────────────────────

class ViolationSeverity(StrEnum):
    """Severity levels for detected violations."""

    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectedViolation(BaseModel):
    """A single behavior violation above threshold."""

    type: str = Field(description="Violation type (e.g. 'smoking').")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score.",
    )


class ViolationAlert(BaseModel):
    """Frontend-ready payload published to ``decision.violation_alert``."""

    track_id: int = Field(description="ByteTrack persistent vehicle ID.")
    frame_id: int = Field(description="Source frame number.")
    license_plate: str | None = Field(
        default=None,
        description="Detected license plate text.",
    )
    violations: list[DetectedViolation] = Field(
        default_factory=list,
        description="List of threshold-exceeding behaviors.",
    )
    severity: ViolationSeverity = Field(
        description="Aggregated severity classification.",
    )
    qod_session_id: str | None = Field(
        default=None,
        description="QoD session ID if network priority was requested.",
    )
    qod_status: str | None = Field(
        default=None,
        description="QoD session status.",
    )
    recommended_action: str = Field(
        default="ALERT_OPERATOR",
        description="Recommended action for the frontend.",
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of the decision.",
    )
