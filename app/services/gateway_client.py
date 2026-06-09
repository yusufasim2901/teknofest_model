"""
Async HTTP client for GSMA Open Gateway APIs with automatic fallback.

The :class:`GatewayClient` attempts the *real* 5G API first.  If the
request times out, returns a 5xx, or raises a connection error, it
transparently retries against the local mock service.  If the mock also
fails, a safe hardcoded default is returned so the caller never blocks
the agent event loop.

Usage::

    async with GatewayClient(settings) as client:
        session = await client.request_qod_session(
            device_ip="10.0.0.1",
            server_ip="10.0.0.2",
            qos_profile="QOS_E",
        )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from app.models.gateway import (
    NumberVerifyResponse,
    QoDSessionResponse,
    QoDSessionStatus,
    QoSProfile,
)

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_REAL_TIMEOUT: float = 3.0      # seconds
_DEFAULT_MOCK_TIMEOUT: float = 2.0      # seconds
_RETRY_BACKOFF: float = 0.5            # seconds before single retry


class GatewayClient:
    """Async client for GSMA Open Gateway with timeout & fallback.

    Fallback cascade::

        Real API  ──(timeout/5xx/error)──▶  Mock API  ──(error)──▶  Hardcoded default

    Parameters:
        real_base_url:  Base URL of the production 5G gateway (from env).
        mock_base_url:  Base URL of the local mock service.
        real_timeout:   Timeout for the real API call (seconds).
        mock_timeout:   Timeout for the mock API call (seconds).
    """

    def __init__(
        self,
        real_base_url: str = "https://api.5g-gateway.example.com",
        mock_base_url: str = "http://localhost:8000/gateway",
        *,
        real_timeout: float = _DEFAULT_REAL_TIMEOUT,
        mock_timeout: float = _DEFAULT_MOCK_TIMEOUT,
    ) -> None:
        self._real_base_url: str = real_base_url.rstrip("/")
        self._mock_base_url: str = mock_base_url.rstrip("/")
        self._real_timeout: float = real_timeout
        self._mock_timeout: float = mock_timeout
        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ────────────────────────────────────────────

    async def __aenter__(self) -> GatewayClient:
        """Open the shared HTTP connection pool."""
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
        logger.info("GatewayClient — HTTP connection pool opened.")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Close the shared HTTP connection pool."""
        await self.close()

    async def close(self) -> None:
        """Explicitly close the underlying ``httpx.AsyncClient``."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("GatewayClient — HTTP connection pool closed.")

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active client or raise if not initialised."""
        if self._client is None:
            msg = (
                "GatewayClient is not initialised. "
                "Use `async with GatewayClient(...) as client:` "
                "or call `__aenter__` explicitly."
            )
            raise RuntimeError(msg)
        return self._client

    # ── QoD Session ──────────────────────────────────────────

    async def request_qod_session(
        self,
        device_ip: str,
        server_ip: str,
        qos_profile: str = "QOS_E",
        duration: int = 300,
    ) -> dict[str, Any]:
        """Request a Quality-on-Demand session with automatic fallback.

        Args:
            device_ip:    IPv4 of the device (UE).
            server_ip:    IPv4 of the application server.
            qos_profile:  CAMARA QoS profile identifier.
            duration:     Session duration in seconds.

        Returns:
            Dict with at least ``session_id``, ``status``, and ``source``
            (``"real"``, ``"mock"``, or ``"fallback"``).
        """
        payload: dict[str, Any] = {
            "device": {"ipv4_address": device_ip},
            "application_server": {"ipv4_address": server_ip},
            "qos_profile": qos_profile,
            "duration": duration,
        }

        # ── Attempt 1: Real API ──────────────────────────────
        result = await self._try_request(
            method="POST",
            url=f"{self._real_base_url}/qod/sessions",
            json_payload=payload,
            timeout=self._real_timeout,
            label="real",
            retry=True,
        )
        if result is not None:
            return result

        # ── Attempt 2: Mock API ──────────────────────────────
        result = await self._try_request(
            method="POST",
            url=f"{self._mock_base_url}/qod/sessions",
            json_payload=payload,
            timeout=self._mock_timeout,
            label="mock",
            retry=False,
        )
        if result is not None:
            return result

        # ── Attempt 3: Hardcoded fallback ────────────────────
        logger.error(
            "GatewayClient — both real and mock APIs failed.  "
            "Returning hardcoded fallback for QoD session."
        )
        return {
            "session_id": None,
            "status": QoDSessionStatus.UNAVAILABLE,
            "qos_profile": qos_profile,
            "source": "fallback",
        }

    # ── Number Verification ──────────────────────────────────

    async def verify_number(
        self,
        phone_number: str,
        hashed_token: str = "default_token",
    ) -> dict[str, Any]:
        """Verify a phone number with automatic fallback.

        Args:
            phone_number:  E.164 phone number.
            hashed_token:  OAuth2 access token hash.

        Returns:
            Dict with ``device_phone_number_verified`` and ``source``.
        """
        payload: dict[str, Any] = {
            "phone_number": phone_number,
            "hashed_token": hashed_token,
        }

        # ── Attempt 1: Real API ──────────────────────────────
        result = await self._try_request(
            method="POST",
            url=f"{self._real_base_url}/number-verification/verify",
            json_payload=payload,
            timeout=self._real_timeout,
            label="real",
            retry=True,
        )
        if result is not None:
            return result

        # ── Attempt 2: Mock API ──────────────────────────────
        result = await self._try_request(
            method="POST",
            url=f"{self._mock_base_url}/number-verification/verify",
            json_payload=payload,
            timeout=self._mock_timeout,
            label="mock",
            retry=False,
        )
        if result is not None:
            return result

        # ── Attempt 3: Hardcoded fallback ────────────────────
        logger.error(
            "GatewayClient — both APIs failed for number verification.  "
            "Returning unverified fallback."
        )
        return {
            "device_phone_number_verified": False,
            "source": "fallback",
        }

    # ── Internal helpers ─────────────────────────────────────

    async def _try_request(
        self,
        *,
        method: str,
        url: str,
        json_payload: dict[str, Any],
        timeout: float,
        label: str,
        retry: bool,
    ) -> dict[str, Any] | None:
        """Attempt an HTTP request with timeout, optional retry, and error handling.

        Returns the parsed JSON response on success, or ``None`` on failure.
        """
        client: httpx.AsyncClient = self._get_client()
        attempts: int = 2 if retry else 1

        for attempt in range(1, attempts + 1):
            t0: float = time.monotonic()
            try:
                response: httpx.Response = await asyncio.wait_for(
                    client.request(
                        method=method,
                        url=url,
                        json=json_payload,
                        timeout=timeout,
                    ),
                    timeout=timeout + 0.5,  # Outer guard
                )
                elapsed_ms: float = (time.monotonic() - t0) * 1000

                if response.status_code < 400:
                    data: dict[str, Any] = response.json()
                    data["source"] = label
                    logger.info(
                        "GatewayClient [%s] — %s %s → %d (%.0f ms).",
                        label,
                        method,
                        url,
                        response.status_code,
                        elapsed_ms,
                    )
                    return data

                # Server error → log and fall through
                logger.warning(
                    "GatewayClient [%s] — %s %s → %d (%.0f ms).  "
                    "Body: %s",
                    label,
                    method,
                    url,
                    response.status_code,
                    elapsed_ms,
                    response.text[:200],
                )

            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "GatewayClient [%s] — timeout after %.0f ms "
                    "(attempt %d/%d): %s",
                    label,
                    elapsed_ms,
                    attempt,
                    attempts,
                    exc,
                )

            except (httpx.ConnectError, httpx.NetworkError, OSError) as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "GatewayClient [%s] — connection error after %.0f ms "
                    "(attempt %d/%d): %s",
                    label,
                    elapsed_ms,
                    attempt,
                    attempts,
                    exc,
                )

            # Backoff before retry
            if attempt < attempts:
                await asyncio.sleep(_RETRY_BACKOFF)

        return None
