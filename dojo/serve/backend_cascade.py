"""Multi-backend serving cascade with automatic fallback (d47).

The inspector flagged d47 with "routing+CB" — the multi-backend wiring was
real but lacked an automatic fallback policy for cross-backend cascades.
This module wraps a list of backends and tries them in order until one
succeeds, emitting per-attempt evidence and a final cascade summary.

Backends only need to expose a ``generate(prompt, **kwargs) -> dict`` API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class BackendCascadeExhausted(DojoError):
    """Raised when all backends in the cascade failed."""


@dataclass
class BackendAttempt:
    """Result of a single backend attempt."""

    backend_name: str
    succeeded: bool
    error: Optional[str] = None
    result: Optional[Dict] = None
    duration_ms: float = 0.0


@dataclass
class BackendCascade:
    """Try a list of backends in order until one succeeds.

    Each backend is a (name, callable) pair where the callable has the
    signature ``f(prompt, **kwargs) -> dict``.  Exceptions are caught and
    the next backend is tried.
    """

    backends: List[Tuple[str, Callable]]
    attempt_log: List[BackendAttempt] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.backends:
            raise BackendCascadeExhausted(
                "BackendCascade requires at least one backend",
                reason="empty_backend_list",
            )

    def generate(self, prompt: str, **kwargs) -> Dict:
        """Try each backend; return the first successful result.

        Raises:
            BackendCascadeExhausted: if all backends failed.  The attached
            ``attempt_log`` records what each backend returned.
        """
        import time
        self.attempt_log = []
        for name, backend in self.backends:
            t0 = time.monotonic()
            try:
                result = backend(prompt, **kwargs)
                duration = (time.monotonic() - t0) * 1000.0
                attempt = BackendAttempt(
                    backend_name=name,
                    succeeded=True,
                    result=result,
                    duration_ms=duration,
                )
                self.attempt_log.append(attempt)
                self._emit_attempt(attempt)
                self._emit_cascade_summary(success=True)
                return result
            except Exception as exc:
                duration = (time.monotonic() - t0) * 1000.0
                attempt = BackendAttempt(
                    backend_name=name,
                    succeeded=False,
                    error=str(exc),
                    duration_ms=duration,
                )
                self.attempt_log.append(attempt)
                self._emit_attempt(attempt)
                continue

        self._emit_cascade_summary(success=False)
        errors = "; ".join(f"{a.backend_name}={a.error}" for a in self.attempt_log)
        raise BackendCascadeExhausted(
            f"all {len(self.backends)} backends failed: {errors}",
            reason=f"{len(self.backends)}_attempts_failed",
        )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def _emit_attempt(self, attempt: BackendAttempt) -> None:
        EvidenceLogger.get_instance().emit(
            {
                "type": "serve.cascade_attempt",
                "data": {
                    "backend": attempt.backend_name,
                    "succeeded": attempt.succeeded,
                    "duration_ms": attempt.duration_ms,
                    "error": attempt.error,
                },
            },
            organ="serve",
        )

    def _emit_cascade_summary(self, success: bool) -> None:
        EvidenceLogger.get_instance().emit(
            {
                "type": "serve.cascade_complete",
                "data": {
                    "success": success,
                    "attempts": len(self.attempt_log),
                    "winning_backend": (
                        self.attempt_log[-1].backend_name
                        if (success and self.attempt_log) else None
                    ),
                    "backends_tried": [a.backend_name for a in self.attempt_log],
                },
            },
            organ="serve",
        )


def first_successful(
    backends: List[Tuple[str, Callable]],
    prompt: str,
    **kwargs,
) -> Tuple[str, Dict]:
    """Convenience: run the cascade once, return (winning_name, result)."""
    cascade = BackendCascade(backends=backends)
    result = cascade.generate(prompt, **kwargs)
    return cascade.attempt_log[-1].backend_name, result


__all__ = [
    "BackendCascade",
    "BackendAttempt",
    "BackendCascadeExhausted",
    "first_successful",
]
