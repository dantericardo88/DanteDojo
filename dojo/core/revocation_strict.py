"""Fail-closed revocation gate (d08).

The inspector flagged d08 with "fails-open on net error" — when the
revocation endpoint was unreachable the existing gate let the operation
proceed.  This module adds a strict wrapper that flips the failure mode:
network errors are treated as revocation by default, so an offline auditor
can't be bypassed by yanking the network.

Two modes:
  - ``StrictRevocationCheck(fail_closed=True)``  → network error = revoked
  - ``StrictRevocationCheck(fail_closed=False)`` → network error = unknown
                                                   (caller decides)
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class PassportRevokedStrict(DojoError):
    """Raised when the strict revocation gate refuses to allow a passport."""


@dataclass(frozen=True)
class RevocationResult:
    """Structured outcome of a revocation check."""

    revoked: bool
    reason: str
    network_error: bool


@dataclass
class StrictRevocationCheck:
    """Fail-closed wrapper around a revocation endpoint.

    Args:
        endpoint_url:  HTTP(S) URL where a HEAD/GET reports the passport's
                       revocation status (HTTP 200 = valid, 404 = revoked).
        fail_closed:   If True (default), a network or 5xx error is treated
                       as revoked.  If False, returns ``RevocationResult``
                       with ``revoked=False, network_error=True`` and the
                       caller decides what to do.
        timeout_s:     Per-request timeout in seconds.
        urlopen:       Override for dependency injection in tests.
    """

    endpoint_url: str
    fail_closed: bool = True
    timeout_s: float = 3.0
    urlopen: Optional[Callable] = None

    def check(self, passport_id: str) -> RevocationResult:
        """Return a RevocationResult — does not raise."""
        url = f"{self.endpoint_url.rstrip('/')}/{passport_id}"
        opener = self.urlopen or urllib.request.urlopen
        try:
            req = urllib.request.Request(url, method="HEAD")
            with opener(req, timeout=self.timeout_s) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= status < 300:
                    result = RevocationResult(revoked=False, reason="ok", network_error=False)
                elif status == 404:
                    result = RevocationResult(revoked=True, reason="not_found", network_error=False)
                else:
                    result = RevocationResult(
                        revoked=self.fail_closed,
                        reason=f"http_{status}",
                        network_error=False,
                    )
        except urllib.error.URLError as exc:
            result = RevocationResult(
                revoked=self.fail_closed,
                reason=f"network_error:{exc.reason}" if hasattr(exc, "reason") else "network_error",
                network_error=True,
            )
        except Exception as exc:
            result = RevocationResult(
                revoked=self.fail_closed,
                reason=f"unexpected:{type(exc).__name__}",
                network_error=True,
            )

        self._emit(passport_id, result)
        return result

    def gate(self, passport_id: str) -> None:
        """Raise ``PassportRevokedStrict`` if the passport is revoked."""
        r = self.check(passport_id)
        if r.revoked:
            raise PassportRevokedStrict(
                f"passport {passport_id} revoked: {r.reason}",
                reason=r.reason,
            )

    def _emit(self, passport_id: str, result: RevocationResult) -> None:
        EvidenceLogger.get_instance().emit(
            {
                "type": "passport.revocation_check_strict",
                "data": {
                    "passport_id": passport_id,
                    "revoked": result.revoked,
                    "reason": result.reason,
                    "network_error": result.network_error,
                    "fail_closed": self.fail_closed,
                },
            },
            organ="core",
        )


__all__ = [
    "StrictRevocationCheck",
    "RevocationResult",
    "PassportRevokedStrict",
]
