"""
Decision Agent — evaluates concatenated vision model results.

Subscribes to ``behavior.results_ready`` on RabbitMQ, applies configurable
violation thresholds, optionally requests a QoD session for critical alerts,
and publishes frontend-ready payloads to ``decision.violation_alert``.

The agent never blocks on 5G network calls: the QoD request is wrapped in
an ``asyncio.wait_for`` with a configurable timeout.  If the network is
unreachable, the :class:`GatewayClient` falls back to the local mock and
ultimately to a safe hardcoded default.

Usage::

    agent = DecisionAgent(
        name="decision_agent",
        event_bus=event_bus,
        gateway_client=gateway_client,
    )
    await agent.start(subscribe_keys=["behavior.results_ready"])
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.models.events import AgentEvent
from app.models.gateway import (
    DetectedViolation,
    ViolationAlert,
    ViolationSeverity,
)
from app.services.gateway_client import GatewayClient

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Threshold Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViolationThresholds:
    """Per-behavior confidence thresholds.

    A behaviour is flagged as a violation when its confidence score
    meets or exceeds the corresponding threshold.
    """

    smoking: float = 0.70
    phone_usage: float = 0.65
    fatigue: float = 0.60

    # Additional custom thresholds can be appended here
    custom: dict[str, float] = field(default_factory=dict)

    def get(self, behavior: str) -> float | None:
        """Look up the threshold for a named behavior.

        Returns ``None`` if the behaviour is unknown (i.e. no threshold set).
        """
        if hasattr(self, behavior):
            return getattr(self, behavior)
        return self.custom.get(behavior)


# ──────────────────────────────────────────────────────────────────────
# Decision Agent
# ──────────────────────────────────────────────────────────────────────

# Default timeout for the QoD network call (seconds)
_QOD_TIMEOUT: float = 3.0

# Routing keys
_SUBSCRIBE_KEY: str = "behavior.results_ready"
_PUBLISH_KEY: str = "decision.violation_alert"


class DecisionAgent(BaseAgent):
    """Agent that evaluates behaviour classification results and publishes alerts.

    Parameters:
        name:            Agent identifier.
        event_bus:       Shared EventBus for pub/sub.
        gateway_client:  Async HTTP client for 5G gateway (with fallback).
        thresholds:      Per-behavior confidence thresholds.
        qod_timeout:     Timeout (s) for the QoD network call.
    """

    def __init__(
        self,
        name: str,
        event_bus: Any,
        gateway_client: GatewayClient,
        *,
        thresholds: ViolationThresholds | None = None,
        qod_timeout: float = _QOD_TIMEOUT,
    ) -> None:
        super().__init__(name=name, event_bus=event_bus)
        self._gateway: GatewayClient = gateway_client
        self._thresholds: ViolationThresholds = thresholds or ViolationThresholds()
        self._qod_timeout: float = qod_timeout

        # Counters for observability
        self._total_events: int = 0
        self._total_violations: int = 0

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self, subscribe_keys: list[str] | None = None) -> None:
        """Start the agent with default subscription if none provided."""
        keys: list[str] = subscribe_keys or [_SUBSCRIBE_KEY]
        await super().start(subscribe_keys=keys)
        self._logger.info(
            "DecisionAgent ready  (thresholds: smoking=%.2f, phone=%.2f, fatigue=%.2f).",
            self._thresholds.smoking,
            self._thresholds.phone_usage,
            self._thresholds.fatigue,
        )

    async def stop(self) -> None:
        """Shutdown hook — log summary statistics."""
        self._logger.info(
            "DecisionAgent shutting down.  "
            "Events processed: %d, violations published: %d.",
            self._total_events,
            self._total_violations,
        )
        await super().stop()

    # ── Core event handler ───────────────────────────────────

    async def on_event(self, event: AgentEvent) -> None:
        """Evaluate a behaviour classification result.

        Expected ``event.payload`` shape::

            {
                "track_id": int,
                "frame_id": int,
                "license_plate": str | None,
                "behaviors": {"smoking": float, "phone_usage": float, "fatigue": float},
                "vehicle_speed_kmh": float | None,
            }
        """
        self._total_events += 1
        payload: dict[str, Any] = event.payload

        # ── Extract fields (defensive) ───────────────────────
        track_id: int = payload.get("track_id", -1)
        frame_id: int = payload.get("frame_id", -1)
        license_plate: str | None = payload.get("license_plate")
        behaviors: dict[str, float] = payload.get("behaviors", {})

        self._logger.debug(
            "Evaluating track_id=%d, frame_id=%d, behaviors=%s.",
            track_id,
            frame_id,
            behaviors,
        )

        # ── Apply thresholds ─────────────────────────────────
        violations: list[DetectedViolation] = self._evaluate_violations(behaviors)

        if not violations:
            self._logger.debug(
                "No violations for track_id=%d, frame_id=%d.",
                track_id,
                frame_id,
            )
            return

        # ── Classify severity ────────────────────────────────
        severity: ViolationSeverity = self._classify_severity(violations)

        # ── QoD enrichment (non-blocking, best-effort) ───────
        qod_session_id: str | None = None
        qod_status: str | None = None

        if severity == ViolationSeverity.CRITICAL:
            qod_result: dict[str, Any] = await self._request_qod_safe(
                device_ip="10.0.0.1",
                server_ip="10.0.0.2",
            )
            qod_session_id = str(qod_result.get("session_id")) if qod_result.get("session_id") else None
            qod_status = qod_result.get("status")
            self._logger.info(
                "QoD session for CRITICAL alert: id=%s, status=%s, source=%s.",
                qod_session_id,
                qod_status,
                qod_result.get("source", "unknown"),
            )

        # ── Build violation alert ────────────────────────────
        alert = ViolationAlert(
            track_id=track_id,
            frame_id=frame_id,
            license_plate=license_plate,
            violations=violations,
            severity=severity,
            qod_session_id=qod_session_id,
            qod_status=qod_status,
            recommended_action=self._recommend_action(severity),
            timestamp_utc=datetime.now(tz=timezone.utc),
        )

        # ── Publish ──────────────────────────────────────────
        await self.publish(
            event_type=_PUBLISH_KEY,
            payload=alert.model_dump(mode="json"),
            correlation_id=str(event.correlation_id) if event.correlation_id else None,
        )

        self._total_violations += 1
        self._logger.info(
            "🚨 Violation alert published — track=%d, frame=%d, "
            "severity=%s, violations=%s.",
            track_id,
            frame_id,
            severity.value,
            [v.type for v in violations],
        )

    # ── Business logic helpers ───────────────────────────────

    def _evaluate_violations(
        self,
        behaviors: dict[str, float],
    ) -> list[DetectedViolation]:
        """Check each behavior against its threshold.

        Args:
            behaviors: Mapping of behavior name → confidence score.

        Returns:
            List of violations that exceed their thresholds.
        """
        violations: list[DetectedViolation] = []

        for behavior_name, confidence in behaviors.items():
            threshold: float | None = self._thresholds.get(behavior_name)
            if threshold is None:
                self._logger.debug(
                    "Unknown behavior '%s' — skipping (no threshold).",
                    behavior_name,
                )
                continue

            if confidence >= threshold:
                violations.append(
                    DetectedViolation(
                        type=behavior_name,
                        confidence=round(confidence, 4),
                    ),
                )

        return violations

    @staticmethod
    def _classify_severity(
        violations: list[DetectedViolation],
    ) -> ViolationSeverity:
        """Determine aggregate severity from the list of violations.

        Rules:
            * ``CRITICAL`` — ≥ 2 violations **or** any confidence ≥ 0.90.
            * ``HIGH``     — 1 violation with confidence ≥ 0.80.
            * ``MEDIUM``   — everything else.
        """
        if not violations:
            return ViolationSeverity.MEDIUM

        max_confidence: float = max(v.confidence for v in violations)

        if len(violations) >= 2 or max_confidence >= 0.90:
            return ViolationSeverity.CRITICAL

        if max_confidence >= 0.80:
            return ViolationSeverity.HIGH

        return ViolationSeverity.MEDIUM

    @staticmethod
    def _recommend_action(severity: ViolationSeverity) -> str:
        """Map severity to a recommended frontend action.

        Returns:
            Action string consumed by the dashboard UI.
        """
        actions: dict[ViolationSeverity, str] = {
            ViolationSeverity.CRITICAL: "EMERGENCY_STOP_ALERT",
            ViolationSeverity.HIGH: "ALERT_OPERATOR",
            ViolationSeverity.MEDIUM: "LOG_AND_MONITOR",
        }
        return actions.get(severity, "LOG_AND_MONITOR")

    # ── QoD integration (with timeout guard) ─────────────────

    async def _request_qod_safe(
        self,
        device_ip: str,
        server_ip: str,
    ) -> dict[str, Any]:
        """Request a QoD session, never blocking longer than ``_qod_timeout``.

        If the entire operation (including fallback) exceeds the timeout,
        a safe default is returned.

        Args:
            device_ip: UE IPv4 address.
            server_ip: Application server IPv4 address.

        Returns:
            QoD session dict (may come from real, mock, or hardcoded fallback).
        """
        try:
            result: dict[str, Any] = await asyncio.wait_for(
                self._gateway.request_qod_session(
                    device_ip=device_ip,
                    server_ip=server_ip,
                    qos_profile="QOS_E",
                    duration=60,
                ),
                timeout=self._qod_timeout,
            )
            return result

        except asyncio.TimeoutError:
            self._logger.warning(
                "QoD request timed out after %.1f s — using fallback.",
                self._qod_timeout,
            )
            return {
                "session_id": None,
                "status": "UNAVAILABLE",
                "source": "timeout_fallback",
            }

        except Exception as exc:
            self._logger.error(
                "QoD request failed unexpectedly: %s — using fallback.",
                exc,
            )
            return {
                "session_id": None,
                "status": "UNAVAILABLE",
                "source": "error_fallback",
            }
