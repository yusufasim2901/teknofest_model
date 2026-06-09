#!/usr/bin/env python3
"""
MAS Infrastructure Health Check.

Verifies the operational status of all critical MAS infrastructure
components: Docker containers, Redis memory limits, RabbitMQ queue
depths, and FastAPI liveness.

Usage:
    python scripts/health_check.py [--json]

Returns:
    Exit code 0 if all checks pass, 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx
import redis

# ──────────────────────────────────────────────────────────────────────
# Config & Defaults
# ──────────────────────────────────────────────────────────────────────

# Docker containers that must be running
_EXPECTED_CONTAINERS = {"mas-rabbitmq", "mas-redis", "mas-postgres", "mas-api"}

# API endpoints (assuming default localhost mappings)
_RABBITMQ_API = "http://localhost:15672/api"
_RABBITMQ_AUTH = ("mas_admin", "changeme_rabbitmq_secret")  # Default from .env.example
_REDIS_URL = "redis://localhost:6379/0"
_FASTAPI_URL = "http://localhost:8000/health"

# Thresholds
_MAX_QUEUE_DEPTH = 1000
_MAX_REDIS_MEMORY_PERCENT = 80.0

# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: bool
    details: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": "PASS" if self.status else "FAIL",
            "details": self.details,
            "error": self.error,
        }

# ──────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────

def check_docker_containers() -> list[CheckResult]:
    """Verify all required Docker containers are running and healthy."""
    results: list[CheckResult] = []
    
    try:
        # Get formatting output from docker ps
        cmd = ["docker", "ps", "--format", "{{.Names}}|{{.State}}|{{.Status}}"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        
        running_containers: dict[str, dict[str, str]] = {}
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                name, state, status = parts[0], parts[1], parts[2]
                running_containers[name] = {"state": state, "status": status}
                
        for expected in _EXPECTED_CONTAINERS:
            if expected not in running_containers:
                results.append(
                    CheckResult(
                        name=f"Docker: {expected}",
                        status=False,
                        details="Container not found or not running.",
                    )
                )
            else:
                info = running_containers[expected]
                is_healthy = "unhealthy" not in info["status"].lower()
                is_running = info["state"].lower() == "running"
                
                results.append(
                    CheckResult(
                        name=f"Docker: {expected}",
                        status=is_running and is_healthy,
                        details=f"{info['state']} ({info['status']})",
                    )
                )
                
    except subprocess.CalledProcessError as e:
        results.append(
            CheckResult(
                name="Docker: daemon",
                status=False,
                details="Failed to execute docker ps",
                error=e.output,
            )
        )
    except FileNotFoundError:
        results.append(
            CheckResult(
                name="Docker: daemon",
                status=False,
                details="Docker executable not found in PATH",
            )
        )

    return results


def check_redis() -> list[CheckResult]:
    """Verify Redis connectivity and memory usage."""
    results: list[CheckResult] = []
    
    try:
        r = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
        
        # Check connectivity
        ping_ok = r.ping()
        results.append(
            CheckResult(
                name="Redis: connectivity",
                status=ping_ok,
                details="PONG" if ping_ok else "No response",
            )
        )
        
        if ping_ok:
            # Check memory
            info = r.info("memory")
            used_human = info.get("used_memory_human", "?")
            
            # Note: Redis info memory maxmemory is 0 if not set, but we set it in docker-compose
            max_bytes = int(info.get("maxmemory", 0))
            used_bytes = int(info.get("used_memory", 0))
            
            if max_bytes > 0:
                percent = (used_bytes / max_bytes) * 100
                max_human = info.get("maxmemory_human", "?")
                
                results.append(
                    CheckResult(
                        name="Redis: memory",
                        status=percent < _MAX_REDIS_MEMORY_PERCENT,
                        details=f"{used_human} / {max_human} ({percent:.1f}%)",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="Redis: memory",
                        status=True,
                        details=f"{used_human} (no limit set)",
                    )
                )
                
    except redis.RedisError as e:
        results.append(
            CheckResult(
                name="Redis: service",
                status=False,
                details="Connection failed",
                error=str(e),
            )
        )

    return results


def check_rabbitmq() -> list[CheckResult]:
    """Verify RabbitMQ queues and connections via Management API."""
    results: list[CheckResult] = []
    
    try:
        with httpx.Client(auth=_RABBITMQ_AUTH, timeout=3.0) as client:
            # Check Queues
            r_queues = client.get(f"{_RABBITMQ_API}/queues")
            r_queues.raise_for_status()
            queues = r_queues.json()
            
            max_depth = 0
            if queues:
                max_depth = max(q.get("messages_ready", 0) for q in queues)
                
            results.append(
                CheckResult(
                    name="RabbitMQ: queues",
                    status=max_depth < _MAX_QUEUE_DEPTH,
                    details=f"{len(queues)} queues, max depth: {max_depth}",
                    error=f"Queue depth exceeds limit ({_MAX_QUEUE_DEPTH})" if max_depth >= _MAX_QUEUE_DEPTH else None,
                )
            )
            
            # Check Connections
            r_conn = client.get(f"{_RABBITMQ_API}/connections")
            r_conn.raise_for_status()
            connections = r_conn.json()
            
            results.append(
                CheckResult(
                    name="RabbitMQ: connections",
                    status=True,  # It's okay to have 0 if idle
                    details=f"{len(connections)} active",
                )
            )
            
    except httpx.HTTPError as e:
        results.append(
            CheckResult(
                name="RabbitMQ: api",
                status=False,
                details="Management API unreachable",
                error=str(e),
            )
        )
        
    return results


def check_fastapi() -> list[CheckResult]:
    """Verify FastAPI liveness probe."""
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(_FASTAPI_URL)
            
            if r.status_code == 200:
                data = r.json()
                status_str = data.get("status", "unknown")
                return [
                    CheckResult(
                        name="FastAPI: /health",
                        status=status_str == "healthy",
                        details=status_str,
                    )
                ]
            else:
                return [
                    CheckResult(
                        name="FastAPI: /health",
                        status=False,
                        details=f"HTTP {r.status_code}",
                        error=r.text,
                    )
                ]
    except httpx.HTTPError as e:
        return [
            CheckResult(
                name="FastAPI: /health",
                status=False,
                details="Endpoint unreachable",
                error=str(e),
            )
        ]


# ──────────────────────────────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MAS Infrastructure Health Check")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    all_results: list[CheckResult] = []
    
    # Run all checks
    all_results.extend(check_docker_containers())
    all_results.extend(check_redis())
    all_results.extend(check_rabbitmq())
    all_results.extend(check_fastapi())
    
    all_passed = all(r.status for r in all_results)
    
    if args.json:
        output = {
            "status": "PASS" if all_passed else "FAIL",
            "checks": [r.to_dict() for r in all_results],
        }
        print(json.dumps(output, indent=2))
        sys.exit(0 if all_passed else 1)
        
    # Text output
    print("━━━ MAS Health Check ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    for r in all_results:
        icon = "✅" if r.status else "❌"
        # Pad the name for alignment
        name_padded = f"{r.name:25}"
        line = f"  {icon} {name_padded} {r.details}"
        if r.error and not r.status:
            line += f"  (Error: {r.error})"
        print(line)
        
    print("\n━━━ Result: " + ("ALL CHECKS PASSED" if all_passed else "FAILURES DETECTED") + " ━━━━━━━━━━━━━━━━━━━━━━")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
