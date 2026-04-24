"""Base FSM class for DanteDojo subsystems.

Every non-trivial stateful operation uses an explicit finite-state machine
that subclasses BaseFSM. Illegal transitions raise FSMTransitionError — no
silent state corruption.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, FrozenSet, Generic, Optional, TypeVar

from dojo.core.exceptions import FSMTransitionError

S = TypeVar("S", bound=Enum)


class BaseFSM(Generic[S]):
    """Thread-unsafe FSM base. Subclasses define allowed transitions and hook on_transition.

    Usage::

        class MyState(Enum):
            IDLE = "idle"
            RUNNING = "running"
            DONE = "done"
            FAILED = "failed"

        class MyFSM(BaseFSM[MyState]):
            _TRANSITIONS = {
                MyState.IDLE: {MyState.RUNNING},
                MyState.RUNNING: {MyState.DONE, MyState.FAILED},
            }
            initial_state = MyState.IDLE
    """

    _TRANSITIONS: dict[S, FrozenSet[S]] = {}
    initial_state: S

    def __init__(self) -> None:
        self._state: S = self.initial_state
        self._history: list[tuple[S, S]] = []

    @property
    def state(self) -> S:
        return self._state

    @property
    def history(self) -> list[tuple[S, S]]:
        return list(self._history)

    def transition(self, target: S, reason: str = "") -> None:
        """Move to *target* state.

        Raises:
            FSMTransitionError: if the transition is not in ``_TRANSITIONS``.
        """
        allowed: FrozenSet[S] = self._TRANSITIONS.get(self._state, frozenset())  # type: ignore[arg-type]
        if target not in allowed:
            raise FSMTransitionError(
                f"FSM {type(self).__name__}: illegal transition {self._state.value!r} → {target.value!r}",
                reason=reason or f"allowed from {self._state.value!r}: {[s.value for s in allowed]}",
            )
        prev = self._state
        self._state = target
        self._history.append((prev, target))
        self.on_transition(prev, target)

    def on_transition(self, previous: S, current: S) -> None:
        """Override to hook into every state change (e.g., emit evidence)."""

    def is_terminal(self) -> bool:
        """True if the FSM has no outgoing transitions from the current state."""
        return not bool(self._TRANSITIONS.get(self._state))  # type: ignore[arg-type]

    def reset(self) -> None:
        """Reset to initial state and clear history."""
        self._state = self.initial_state
        self._history.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(state={self._state.value!r})"


def fsm_guard(
    allowed_states: FrozenSet[S],
    error_msg: str = "Operation not allowed in current state",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that raises FSMTransitionError if ``self.state`` is not in *allowed_states*."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if self.state not in allowed_states:
                raise FSMTransitionError(
                    error_msg,
                    reason=f"state={self.state.value!r} not in {[s.value for s in allowed_states]}",
                )
            return fn(self, *args, **kwargs)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


__all__ = ["BaseFSM", "fsm_guard"]
