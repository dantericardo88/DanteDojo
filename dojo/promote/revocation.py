"""Sovereign revocation check — HTTP GET to revocation_url at model load."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import PassportRevoked
from dojo.core.one_door import one_door
from dojo.core.types import ConstitutionalPassport


# ---------------------------------------------------------------------------
# In-process revocation cache
# ---------------------------------------------------------------------------

# Cache stores (revoked: bool, timestamp: float) keyed by passport_id.
# Thread-safe via _CACHE_LOCK.
_REVOCATION_CACHE: Dict[str, Tuple[bool, float]] = {}
_CACHE_LOCK = threading.Lock()

# Cache TTL in seconds (0 = cache forever within process lifetime, suitable for tests).
_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes


def _cache_get(passport_id: str) -> Optional[bool]:
    """Return cached revocation status or None if not cached / expired."""
    with _CACHE_LOCK:
        entry = _REVOCATION_CACHE.get(passport_id)
        if entry is None:
            return None
        revoked, ts = entry
        if _CACHE_TTL_SECONDS > 0 and (time.monotonic() - ts) > _CACHE_TTL_SECONDS:
            del _REVOCATION_CACHE[passport_id]
            return None
        return revoked


def _cache_set(passport_id: str, revoked: bool) -> None:
    """Store revocation status in cache."""
    with _CACHE_LOCK:
        _REVOCATION_CACHE[passport_id] = (revoked, time.monotonic())


def _cache_clear() -> None:
    """Clear the entire revocation cache (test helper)."""
    with _CACHE_LOCK:
        _REVOCATION_CACHE.clear()


# ---------------------------------------------------------------------------
# check_revocation
# ---------------------------------------------------------------------------

@one_door("promote.revocation.check")
def check_revocation(
    passport: ConstitutionalPassport,
    timeout_seconds: float = 5.0,
) -> bool:
    """Check whether a passport has been revoked via HTTP GET to revocation_url.

    Emits promote.revocation.checked evidence with passport_id, revoked, and
    latency_ms on every check.  Emits promote.revocation_cache_hit when the
    result is served from the in-process cache (no HTTP call).

    Args:
        passport: ConstitutionalPassport with revocation_url field.
        timeout_seconds: HTTP timeout for the revocation check.

    Returns:
        False (not revoked) — never returns True; raises PassportRevoked instead.

    Raises:
        ValueError: If passport_id is empty.
        PassportRevoked: if the revocation endpoint returns {"revoked": true}.
    """
    logger = EvidenceLogger.get_instance()
    revocation_url = passport.revocation_url
    passport_id = passport.passport_id

    # --- Validate passport_id ---
    if not passport_id:
        raise ValueError("passport_id must not be empty")

    # --- Handle None / missing revocation_url ---
    if not revocation_url:
        logger.emit(
            {
                "type": "promote.revocation.checked",
                "data": {
                    "passport_id": passport_id,
                    "revoked": False,
                    "latency_ms": 0.0,
                    "status": "no_url_skip",
                },
            },
            organ="promote",
        )
        return False

    # --- Cache hit path ---
    cached = _cache_get(passport_id)
    if cached is not None:
        logger.emit(
            {
                "type": "promote.revocation_cache_hit",
                "data": {
                    "passport_id": passport_id,
                    "revoked": cached,
                },
            },
            organ="promote",
        )
        logger.emit(
            {
                "type": "promote.revocation.checked",
                "data": {
                    "passport_id": passport_id,
                    "revoked": cached,
                    "latency_ms": 0.0,
                    "status": "cache_hit",
                },
            },
            organ="promote",
        )
        if cached:
            raise PassportRevoked(
                f"Passport {passport_id!r} has been revoked (cached)",
                reason=f"revocation_url={revocation_url} previously returned {{revoked: true}}",
            )
        return False

    # --- HTTP check ---
    t_start = time.monotonic()

    try:
        req = urllib.request.Request(
            f"{revocation_url}?passport_id={passport_id}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            revoked = bool(data.get("revoked", False))

        latency_ms = (time.monotonic() - t_start) * 1000.0

    except urllib.error.HTTPError as exc:
        latency_ms = (time.monotonic() - t_start) * 1000.0
        # HTTP error codes (404, 500, etc.) → controlled failure
        logger.emit(
            {
                "type": "promote.revocation.checked",
                "data": {
                    "passport_id": passport_id,
                    "revoked": False,
                    "latency_ms": latency_ms,
                    "status": f"http_error_{exc.code}",
                    "http_status": exc.code,
                },
            },
            organ="promote",
        )
        raise PassportRevoked(
            f"Passport {passport_id!r} revocation check failed with HTTP {exc.code}",
            reason=f"revocation_url={revocation_url} returned HTTP {exc.code}",
        ) from exc

    except urllib.error.URLError:
        latency_ms = (time.monotonic() - t_start) * 1000.0
        # Network failure: fail-open (allow) but log
        logger.emit(
            {
                "type": "promote.revocation.checked",
                "data": {
                    "passport_id": passport_id,
                    "revoked": False,
                    "latency_ms": latency_ms,
                    "status": "network_failure_allow",
                    "reason": "revocation endpoint unreachable",
                },
            },
            organ="promote",
        )
        return False

    except (json.JSONDecodeError, KeyError):
        latency_ms = (time.monotonic() - t_start) * 1000.0
        # Malformed response: fail-open
        logger.emit(
            {
                "type": "promote.revocation.checked",
                "data": {
                    "passport_id": passport_id,
                    "revoked": False,
                    "latency_ms": latency_ms,
                    "status": "parse_error_allow",
                },
            },
            organ="promote",
        )
        return False

    # Store result in cache
    _cache_set(passport_id, revoked)

    # Emit final evidence
    logger.emit(
        {
            "type": "promote.revocation.checked",
            "data": {
                "passport_id": passport_id,
                "revocation_url": revocation_url,
                "revoked": revoked,
                "latency_ms": latency_ms,
                "status": "revoked" if revoked else "valid",
            },
        },
        organ="promote",
    )

    if revoked:
        raise PassportRevoked(
            f"Passport {passport_id!r} has been revoked",
            reason=f"revocation_url={revocation_url} returned {{revoked: true}}",
        )

    return False


__all__ = ["check_revocation", "_cache_clear"]
