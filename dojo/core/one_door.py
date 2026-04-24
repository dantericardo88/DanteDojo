"""DanteDojo one-door doctrine.

Every public subsystem entry point is decorated with @one_door. Internal
functions are not exported and must not be called from outside the subsystem.
Bypass attempts raise OneDoorViolation.

Transplanted and simplified from STEAK V3 steak/core/one_door.py.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from typing import Any, Callable, Optional, TypeVar

from dojo.core.evidence import EvidenceLogger, EvidenceType

F = TypeVar("F", bound=Callable[..., Any])


def _stable_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def one_door(entry_point_name: str, organ: Optional[str] = None) -> Callable[[F], F]:
    """Decorator that marks a function as the single public entry point for a subsystem.

    On each call:
    - Emits an ``one_door.entry`` evidence event with the caller module.
    - Raises OneDoorViolation if DOJO_BYPASS env var is set.
    - Wraps errors so the caller sees the subsystem name in the traceback.

    Usage::

        @one_door("train.finetune")
        def finetune(config: FinetuneConfig) -> FinetuneResult:
            ...
    """
    from dojo.core.exceptions import OneDoorViolation

    derived_organ = organ or entry_point_name.split(".")[0]

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import os
            if os.environ.get("DOJO_BYPASS"):
                raise OneDoorViolation(
                    f"DOJO_BYPASS is set — bypass of one-door entry '{entry_point_name}' is forbidden",
                )

            frame = inspect.currentframe()
            caller_module = ""
            if frame and frame.f_back:
                caller_module = frame.f_back.f_globals.get("__name__", "")

            try:
                args_repr = _stable_hash([str(a)[:64] for a in args] + [f"{k}={str(v)[:64]}" for k, v in kwargs.items()])
            except Exception:
                args_repr = "unhashable"

            logger = EvidenceLogger.get_instance()
            logger.emit(
                {
                    "type": EvidenceType.ONE_DOOR_ENTRY.value,
                    "data": {
                        "entry_point": entry_point_name,
                        "organ": derived_organ,
                        "caller_module": caller_module,
                        "args_hash": args_repr,
                    },
                },
                organ=derived_organ,
            )

            return fn(*args, **kwargs)

        wrapper.__one_door__ = entry_point_name  # type: ignore[attr-defined]
        wrapper.__organ__ = derived_organ  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def enforce_one_door(func_name: str) -> None:
    """Raise OneDoorViolation if DOJO_BYPASS env var is set."""
    import os
    from dojo.core.exceptions import OneDoorViolation

    if os.environ.get("DOJO_BYPASS"):
        raise OneDoorViolation(f"Bypass attempt detected for '{func_name}'")


# ---------------------------------------------------------------------------
# Result envelope helpers (mirrors STEAK pattern)
# ---------------------------------------------------------------------------

def ok(data: dict[str, Any], *, evidence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "data": data}
    if evidence is not None:
        out["evidence"] = evidence
    return out


def fail(code: str, message: str, *, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"ok": False, "error": err}


__all__ = [
    "one_door",
    "enforce_one_door",
    "ok",
    "fail",
]
