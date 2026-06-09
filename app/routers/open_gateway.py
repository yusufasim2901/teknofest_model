"""
Mock GSMA Open Gateway endpoints for local development.

Implements dummy ``Quality on Demand`` (QoD) and ``Number Verification``
APIs that mirror the CAMARA specification closely enough for integration
testing.  Responses include randomized latency simulation and controlled
failure injection so downstream consumers (e.g. the ``GatewayClient``)
can be tested against realistic network behaviour.

Mount this router on the main FastAPI app::

    from app.routers.open_gateway import router as gateway_router
    app.include_router(gateway_router)
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Header, Response, status

from app.models.gateway import (
    DeviceIdentifier,
    GatewayErrorResponse,
    NumberVerifyRequest,
    NumberVerifyResponse,
    QoDSessionRequest,
    QoDSessionResponse,
    QoDSessionStatus,
)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger(__name__)

# Simulated latency bounds (seconds)
_QOD_LATENCY_MIN: float = 0.05
_QOD_LATENCY_MAX: float = 0.30
_NUMVERIFY_LATENCY_MIN: float = 0.03
_NUMVERIFY_LATENCY_MAX: float = 0.15

# Probability of injecting a 503 on QoD create (0.0–1.0)
_QOD_FAILURE_RATE: float = 0.10

# In-memory session store (mock — no persistence needed)
_sessions: dict[str, QoDSessionResponse] = {}

# ──────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/gateway",
    tags=["5G Mock Gateway"],
)


def _correlator_header() -> dict[str, str]:
    """Generate the CAMARA ``x-correlator`` response header."""
    return {"x-correlator": str(uuid4())}


# ──────────────────────────────────────────────────────────────────────
# QoD Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/qod/sessions",
    response_model=QoDSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a QoD session (mock)",
    responses={
        503: {"model": GatewayErrorResponse, "description": "Simulated outage"},
    },
)
async def create_qod_session(
    body: QoDSessionRequest,
    response: Response,
) -> QoDSessionResponse:
    """Create a mock Quality-on-Demand session.

    Simulates real-world behavior:

    * Randomized latency between 50–300 ms.
    * 10 % chance of returning ``503 Service Unavailable``.
    """
    # ── Simulate network latency (non-blocking) ──────────────
    latency: float = random.uniform(_QOD_LATENCY_MIN, _QOD_LATENCY_MAX)
    await asyncio.sleep(latency)

    # ── Inject random failure ────────────────────────────────
    if random.random() < _QOD_FAILURE_RATE:
        logger.warning(
            "Mock QoD — injecting 503 failure (latency=%.0f ms).",
            latency * 1000,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": 503,
                "code": "SERVICE_UNAVAILABLE",
                "message": "QoD service is temporarily unavailable (simulated).",
            },
        )

    # ── Build session ────────────────────────────────────────
    now: datetime = datetime.now(tz=timezone.utc)
    session = QoDSessionResponse(
        session_id=uuid4(),
        status=QoDSessionStatus.AVAILABLE,
        qos_profile=body.qos_profile,
        device=body.device,
        application_server=body.application_server,
        duration=body.duration,
        started_at=now,
        expires_at=now + timedelta(seconds=body.duration),
    )

    # Store in memory
    session_key: str = str(session.session_id)
    _sessions[session_key] = session

    # CAMARA headers
    for key, value in _correlator_header().items():
        response.headers[key] = value

    logger.info(
        "Mock QoD — session created: %s (profile=%s, latency=%.0f ms).",
        session_key,
        body.qos_profile.value,
        latency * 1000,
    )
    return session


@router.get(
    "/qod/sessions/{session_id}",
    response_model=QoDSessionResponse,
    summary="Retrieve a QoD session (mock)",
    responses={
        404: {"model": GatewayErrorResponse, "description": "Session not found"},
    },
)
async def get_qod_session(
    session_id: UUID,
    response: Response,
) -> QoDSessionResponse:
    """Retrieve an existing mock QoD session by ID."""
    session_key: str = str(session_id)
    session: QoDSessionResponse | None = _sessions.get(session_key)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": 404,
                "code": "NOT_FOUND",
                "message": f"Session '{session_key}' does not exist.",
            },
        )

    for key, value in _correlator_header().items():
        response.headers[key] = value

    return session


@router.delete(
    "/qod/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a QoD session (mock)",
    responses={
        404: {"model": GatewayErrorResponse, "description": "Session not found"},
    },
)
async def delete_qod_session(
    session_id: UUID,
    response: Response,
) -> None:
    """Delete (terminate) a mock QoD session."""
    session_key: str = str(session_id)

    if session_key not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": 404,
                "code": "NOT_FOUND",
                "message": f"Session '{session_key}' does not exist.",
            },
        )

    del _sessions[session_key]

    for key, value in _correlator_header().items():
        response.headers[key] = value

    logger.info("Mock QoD — session deleted: %s.", session_key)


# ──────────────────────────────────────────────────────────────────────
# Number Verification Endpoints
# ──────────────────────────────────────────────────────────────────────


@router.post(
    "/number-verification/verify",
    response_model=NumberVerifyResponse,
    summary="Verify a phone number (mock)",
    responses={
        401: {"model": GatewayErrorResponse, "description": "Unauthorized"},
    },
)
async def verify_number(
    body: NumberVerifyRequest,
    response: Response,
    authorization: str | None = Header(default=None),
) -> NumberVerifyResponse:
    """Mock number verification endpoint.

    Behaviour:

    * Returns ``verified = True`` for phone numbers ending in an even digit.
    * Returns ``verified = False`` for odd-ending numbers.
    * Simulates ``401 Unauthorized`` if ``hashed_token`` is empty.
    * Randomized latency between 30–150 ms.
    """
    # ── Simulate network latency ─────────────────────────────
    latency: float = random.uniform(
        _NUMVERIFY_LATENCY_MIN,
        _NUMVERIFY_LATENCY_MAX,
    )
    await asyncio.sleep(latency)

    # ── Auth simulation ──────────────────────────────────────
    if not body.hashed_token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": 401,
                "code": "UNAUTHENTICATED",
                "message": "Missing or invalid authentication token.",
            },
        )

    # ── Deterministic verification logic ─────────────────────
    last_digit: str = body.phone_number.rstrip()[-1]
    verified: bool = last_digit.isdigit() and int(last_digit) % 2 == 0

    result = NumberVerifyResponse(
        device_phone_number_verified=verified,
        server_correlation_id=uuid4(),
    )

    for key, value in _correlator_header().items():
        response.headers[key] = value

    logger.info(
        "Mock NumVerify — number=%s…%s, verified=%s (latency=%.0f ms).",
        body.phone_number[:4],
        body.phone_number[-2:],
        verified,
        latency * 1000,
    )
    return result


# ──────────────────────────────────────────────────────────────────────
# Utility — Session Store (for testing)
# ──────────────────────────────────────────────────────────────────────


def get_active_sessions() -> dict[str, QoDSessionResponse]:
    """Return the in-memory session store (for introspection / tests)."""
    return _sessions


def clear_sessions() -> None:
    """Flush all mock sessions (useful in test fixtures)."""
    _sessions.clear()
