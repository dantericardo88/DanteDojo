"""Top-level inference router for DanteDojo.

Routes inference requests according to a RoutingPolicy, enforces
privacy tiers, and records cost receipts.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import (
    NoRouteAvailable,
    PrivacyTierViolation,
    RoutingProviderUnavailable,
)
from dojo.core.one_door import one_door
from dojo.core.types import (
    CostReceipt,
    PrivacyTier,
    RoutingOption,
    RoutingPolicy,
)

from dojo.route.specialist_registry import SpecialistRegistry

# Module-level specialist registry. When set via configure_specialist_registry(),
# _is_viable consults it for capability-aware filtering of LOCAL_SPECIALIST routes.
_active_specialist_registry: Optional[SpecialistRegistry] = None


def configure_specialist_registry(registry: Optional[SpecialistRegistry]) -> None:
    """Install or clear the active SpecialistRegistry used during dispatch."""
    global _active_specialist_registry
    _active_specialist_registry = registry


def get_active_specialist_registry() -> Optional[SpecialistRegistry]:
    return _active_specialist_registry


# Module-level HealthMonitor. start_health_monitor() lazily creates one and
# starts the background probe loop; stop_health_monitor() shuts it down.
# Wired through here so production code (or app startup) only needs to call
# one function instead of constructing the class directly.
_active_health_monitor: Optional["HealthMonitor"] = None  # noqa: F821 forward


def start_health_monitor(
    interval_seconds: float = 30.0,
    probe_timeout_seconds: float = 3.0,
) -> "HealthMonitor":  # noqa: F821
    """Start the singleton HealthMonitor if not already running."""
    global _active_health_monitor
    if _active_health_monitor is None:
        _active_health_monitor = HealthMonitor(
            interval_seconds=interval_seconds,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        _active_health_monitor.start()
    return _active_health_monitor


def stop_health_monitor() -> None:
    """Stop the singleton HealthMonitor and clear it."""
    global _active_health_monitor
    if _active_health_monitor is not None:
        _active_health_monitor.stop()
        _active_health_monitor = None


def get_active_health_monitor() -> Optional["HealthMonitor"]:  # noqa: F821
    return _active_health_monitor


# Providers marked as dead — populated by tests or a health monitor.
# Any RoutingOption.value string placed here will be treated as unavailable.
_DEAD_PROVIDERS: set[str] = set()

# Circuit breaker: tracks consecutive failures per provider.
# After _CIRCUIT_BREAKER_THRESHOLD consecutive failures the provider is added
# to _DEAD_PROVIDERS and the counter resets.
_circuit_breaker: dict[str, int] = {}
_CIRCUIT_BREAKER_THRESHOLD = 3

# Half-open state: providers that are in a trial/probe state after being dead.
# A provider in _HALF_OPEN_PROVIDERS is allowed ONE test call. If it succeeds,
# the provider is restored; if it fails, it goes back to dead.
_HALF_OPEN_PROVIDERS: set[str] = set()

# Timestamps recording when each provider was added to _DEAD_PROVIDERS.
# Used by _is_provider_alive() to auto-promote to half-open after the TTL.
_DEAD_PROVIDER_TIMESTAMPS: dict[str, float] = {}
_RECOVERY_TTL_SECONDS: float = 300.0   # 5 minutes default
_RECOVERY_BACKOFF_MULTIPLIER: float = 1.5
_RECOVERY_ATTEMPTS: dict[str, int] = {}

# Latency-based circuit breaker: tracks recent call latencies per provider.
# If the last _LATENCY_BREAKER_WINDOW consecutive calls each exceed
# _LATENCY_BREAKER_THRESHOLD_MS milliseconds, the provider is tripped.
_latency_window: dict[str, deque] = {}
_LATENCY_BREAKER_THRESHOLD_MS: float = 5000.0   # ms before a call is "slow"
_LATENCY_BREAKER_WINDOW: int = 3                 # consecutive slow calls to trip

# Route options that require external API access
_EXTERNAL_API_OPTIONS: frozenset[RoutingOption] = frozenset({
    RoutingOption.API_HAIKU,
    RoutingOption.API_SONNET,
    RoutingOption.API_DEEPSEEK,
    RoutingOption.API_GPT5_MINI,
})

# Route options satisfied by a local server
_LOCAL_OPTIONS: frozenset[RoutingOption] = frozenset({
    RoutingOption.LOCAL_SPECIALIST,
    RoutingOption.LOCAL_GENERALIST,
})

# Stub cost estimates per external API route (USD per 1k tokens)
_API_COST_PER_1K: dict[RoutingOption, float] = {
    RoutingOption.API_HAIKU: 0.00025,
    RoutingOption.API_SONNET: 0.003,
    RoutingOption.API_DEEPSEEK: 0.00014,
    RoutingOption.API_GPT5_MINI: 0.00015,
}


def _health_check_with_timeout(endpoint: str, timeout: float = 3.0) -> bool:
    """Perform a real HEAD request to *endpoint* with the given timeout.

    Returns True if the server responds with any HTTP status, False on URLError.
    Only called when DOJO_REAL_GPU=1 or DOJO_HEALTH_CHECK=1 is set.
    """
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(endpoint, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        return True
    except urllib.error.URLError:
        return False


def _record_provider_failure(option: RoutingOption) -> None:
    """Increment failure count for *option*; mark dead after threshold.

    Emits ``route.circuit_trip`` evidence when the breaker trips.
    Stamps _DEAD_PROVIDER_TIMESTAMPS so the TTL-based auto-half-open path
    in _is_provider_alive() knows when to re-probe.
    """
    key = option.value
    _circuit_breaker[key] = _circuit_breaker.get(key, 0) + 1
    if _circuit_breaker[key] >= _CIRCUIT_BREAKER_THRESHOLD:
        _DEAD_PROVIDERS.add(key)
        _HALF_OPEN_PROVIDERS.discard(key)
        _DEAD_PROVIDER_TIMESTAMPS[key] = time.monotonic()
        _circuit_breaker[key] = 0
        EvidenceLogger.get_instance().emit(
            {
                "type": "route.circuit_trip",
                "data": {
                    "provider": key,
                    "threshold": _CIRCUIT_BREAKER_THRESHOLD,
                    "reason": "consecutive_failures",
                    "recovery_ttl_seconds": _RECOVERY_TTL_SECONDS,
                },
            },
            organ="route",
        )


def _record_provider_success(option: RoutingOption) -> None:
    """Reset circuit-breaker counter on success; remove from dead/half-open set.

    Emits ``route.circuit_reset`` evidence when a tripped or half-open breaker recovers.
    """
    key = option.value
    was_tripped = key in _DEAD_PROVIDERS or key in _HALF_OPEN_PROVIDERS
    _circuit_breaker[key] = 0
    _DEAD_PROVIDERS.discard(key)
    _HALF_OPEN_PROVIDERS.discard(key)
    _DEAD_PROVIDER_TIMESTAMPS.pop(key, None)
    _RECOVERY_ATTEMPTS.pop(key, None)
    # Also clear any latency window so slow-call history doesn't linger
    if key in _latency_window:
        _latency_window[key].clear()
    if was_tripped:
        EvidenceLogger.get_instance().emit(
            {
                "type": "route.circuit_reset",
                "data": {"provider": key},
            },
            organ="route",
        )


def _record_provider_latency(option: RoutingOption, latency_ms: float) -> None:
    """Record a call latency for *option*.

    If the last _LATENCY_BREAKER_WINDOW consecutive calls all exceeded
    _LATENCY_BREAKER_THRESHOLD_MS, the provider is tripped as dead.
    """
    key = option.value
    if key not in _latency_window:
        _latency_window[key] = deque(maxlen=_LATENCY_BREAKER_WINDOW)
    _latency_window[key].append(latency_ms)

    window = _latency_window[key]
    if (
        len(window) >= _LATENCY_BREAKER_WINDOW
        and all(ms > _LATENCY_BREAKER_THRESHOLD_MS for ms in window)
    ):
        _DEAD_PROVIDERS.add(key)
        _latency_window[key].clear()
        EvidenceLogger.get_instance().emit(
            {
                "type": "route.circuit_trip",
                "data": {
                    "provider": key,
                    "threshold": _CIRCUIT_BREAKER_THRESHOLD,
                    "reason": "consecutive_high_latency",
                    "latency_ms": latency_ms,
                    "window": _LATENCY_BREAKER_WINDOW,
                    "threshold_ms": _LATENCY_BREAKER_THRESHOLD_MS,
                },
            },
            organ="route",
        )


def _enter_half_open(option: RoutingOption) -> None:
    """Move a dead provider into half-open state (one probe call allowed)."""
    key = option.value
    if key in _DEAD_PROVIDERS:
        _DEAD_PROVIDERS.discard(key)
        _HALF_OPEN_PROVIDERS.add(key)


def _is_half_open(option: RoutingOption) -> bool:
    """Return True if the provider is currently in half-open state."""
    return option.value in _HALF_OPEN_PROVIDERS


def _is_provider_alive(option: RoutingOption) -> bool:
    """Return False if the provider is known dead; True otherwise.

    Auto-promotes a dead provider to half-open once the recovery TTL has
    elapsed since it was marked dead. The TTL grows exponentially by
    _RECOVERY_BACKOFF_MULTIPLIER on each successive failure to prevent
    thrashing.

    In real-GPU mode (DOJO_REAL_GPU=1) or when DOJO_HEALTH_CHECK=1 a HEAD
    request is made to the endpoint. In plain stub/test mode only
    _DEAD_PROVIDERS is consulted (no network calls).
    """
    key = option.value
    if key in _DEAD_PROVIDERS:
        dead_at = _DEAD_PROVIDER_TIMESTAMPS.get(key)
        if dead_at is not None:
            attempt = _RECOVERY_ATTEMPTS.get(key, 0)
            ttl = _RECOVERY_TTL_SECONDS * (_RECOVERY_BACKOFF_MULTIPLIER ** attempt)
            if (time.monotonic() - dead_at) >= ttl:
                _DEAD_PROVIDERS.discard(key)
                _HALF_OPEN_PROVIDERS.add(key)
                _RECOVERY_ATTEMPTS[key] = attempt + 1
                EvidenceLogger.get_instance().emit(
                    {
                        "type": "route.circuit_half_open",
                        "data": {
                            "provider": key,
                            "ttl_elapsed_s": time.monotonic() - dead_at,
                            "attempt": attempt + 1,
                        },
                    },
                    organ="route",
                )
                return True
        return False

    real_gpu = os.environ.get("DOJO_REAL_GPU") == "1"
    health_check = os.environ.get("DOJO_HEALTH_CHECK") == "1"

    if real_gpu or health_check:
        endpoint = _api_endpoint(option)
        if endpoint:
            alive = _health_check_with_timeout(endpoint, timeout=3.0)
            if not alive:
                _record_provider_failure(option)
                return False
            _record_provider_success(option)
    return True


@one_door("route.dispatch")
def dispatch(
    task: str,
    input_text: str,
    policy: RoutingPolicy,
    *,
    available_specialists: Optional[list[str]] = None,
    cost_tracker: Optional[Any] = None,
) -> dict:
    """Route an inference request according to the RoutingPolicy.

    Privacy enforcement:
      - CONFIDENTIAL or SECRETS + external API route → PrivacyTierViolation

    Args:
        task: Dante task identifier (e.g. "bookkeeping").
        input_text: The prompt / input text for routing decisions.
        policy: RoutingPolicy describing preferences and constraints.
        available_specialists: List of specialist model names currently
            deployed. Determines whether LOCAL_SPECIALIST is viable.
        cost_tracker: Optional CostTracker to persist cost receipts and
            enforce monthly budget limits.

    Returns:
        dict with keys:
          - route: chosen RoutingOption value string
          - endpoint_url: str or None (None for local when server not started)
          - reason: human-readable rationale
          - cost_receipt: CostReceipt for the dispatch

    Raises:
        PrivacyTierViolation: when CONFIDENTIAL/SECRETS data would reach an API.
        NoRouteAvailable: when no viable route exists.
        RoutingProviderUnavailable: when all preferred external providers are dead
            and no local fallback exists in the preference order.
        BudgetExceeded: when the estimated cost would exceed monthly_limit_usd.
    """
    logger = EvidenceLogger.get_instance()
    specialists = set(available_specialists or [])

    chosen: Optional[RoutingOption] = None
    reason: str = ""
    dead_providers_skipped: list[str] = []

    for option in policy.preference_order:
        viable, why = _is_viable(option, policy, specialists)
        if not viable:
            # Check if this is a privacy violation — raise immediately
            if _is_privacy_violation(option, policy):
                logger.emit(
                    {
                        "type": EvidenceType.ROUTE_PRIVACY_BLOCKED.value,
                        "data": {
                            "task": task,
                            "blocked_option": option.value,
                            "privacy_tier": policy.privacy_tier.value,
                            "reason": why,
                        },
                    },
                    organ="route",
                )
                raise PrivacyTierViolation(
                    f"Privacy tier {policy.privacy_tier.value!r} forbids "
                    f"routing to external API ({option.value})",
                    reason=why,
                )
            # Track dead external providers for RoutingProviderUnavailable logic
            if (
                option in _EXTERNAL_API_OPTIONS
                and "currently unavailable" in why
            ):
                dead_providers_skipped.append(option.value)
                logger.emit(
                    {
                        "type": EvidenceType.ROUTE_FALLBACK.value,
                        "data": {
                            "skipped_provider": option.value,
                            "reason": "provider_unavailable",
                            "task": task,
                        },
                    },
                    organ="route",
                )
                # Emit per-hop fallback evidence (multi-hop tracking)
                logger.emit(
                    {
                        "type": "route.fallback_hop",
                        "data": {
                            "task": task,
                            "failed_provider": option.value,
                            "hop": len(dead_providers_skipped),
                            "reason": why,
                        },
                    },
                    organ="route",
                )
            continue

        chosen = option
        reason = why
        break

    if chosen is None:
        # Emit route.exhausted evidence before raising
        logger.emit(
            {
                "type": "route.exhausted",
                "data": {
                    "task": task,
                    "failed_providers": dead_providers_skipped,
                    "preference_order": [o.value for o in policy.preference_order],
                },
            },
            organ="route",
        )
        # If we skipped dead providers and no local option was available, raise
        # a more specific RoutingProviderUnavailable rather than NoRouteAvailable.
        if dead_providers_skipped:
            raise RoutingProviderUnavailable(
                f"All preferred providers are unavailable for task={task!r}: "
                f"{dead_providers_skipped}",
                reason="no fallback route available after dead provider(s) skipped",
            )
        raise NoRouteAvailable(
            f"No viable route for task={task!r} with policy preferences "
            f"{[o.value for o in policy.preference_order]}",
            reason="all options exhausted or unavailable",
        )

    endpoint_url: Optional[str] = None
    if chosen in _EXTERNAL_API_OPTIONS:
        endpoint_url = _api_endpoint(chosen)

    receipt = _build_receipt(task, chosen, input_text)

    # Budget enforcement before committing the route
    if (
        cost_tracker is not None
        and hasattr(cost_tracker, "monthly_limit_usd")
        and cost_tracker.monthly_limit_usd
    ):
        estimated = _estimate_cost(chosen, input_text)
        cost_tracker.check_budget(estimated)

    logger.emit(
        {
            "type": EvidenceType.ROUTE_DECISION.value,
            "data": {
                "task": task,
                "route": chosen.value,
                "provider_name": _provider_name(chosen),
                "endpoint_url": endpoint_url,
                "reason": reason,
                "privacy_tier": policy.privacy_tier.value,
                "providers_available": [o.value for o in policy.preference_order],
            },
        },
        organ="route",
    )

    if cost_tracker is not None:
        cost_tracker.record(receipt)

    # Sign the cost receipt so downstream auditors can verify the dispatch
    # without trusting the cost_tracker JSONL log. See dojo/route/cost_receipts.py.
    from dojo.route.cost_receipts import build_cost_receipt, sign_cost_receipt
    signed_receipt = sign_cost_receipt(build_cost_receipt(
        request_id=str(getattr(receipt, "receipt_id", "") or _receipt_uuid()),
        provider=_provider_name(chosen),
        prompt_tokens=int(getattr(receipt, "input_tokens", 0)),
        completion_tokens=int(getattr(receipt, "output_tokens", 0)),
        usd_cost=float(getattr(receipt, "estimated_cost_usd", 0.0)),
        privacy_tier=policy.privacy_tier.value,
    ))

    return {
        "route": chosen.value,
        "endpoint_url": endpoint_url,
        "reason": reason,
        "cost_receipt": receipt,
        "signed_cost_receipt": signed_receipt,
    }


def _receipt_uuid() -> str:
    import uuid
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _provider_name(option: RoutingOption) -> str:
    """Return the canonical provider name for an evidence record."""
    return {
        RoutingOption.API_HAIKU: "anthropic",
        RoutingOption.API_SONNET: "anthropic",
        RoutingOption.API_DEEPSEEK: "deepseek",
        RoutingOption.API_GPT5_MINI: "openai",
        RoutingOption.LOCAL_SPECIALIST: "local",
        RoutingOption.LOCAL_GENERALIST: "local",
    }.get(option, "unknown")


def _estimate_cost(option: RoutingOption, input_text: str) -> float:
    """Estimate cost in USD for routing the given input to the given option."""
    token_estimate = max(1, len(input_text.split()))
    cost_per_1k = _API_COST_PER_1K.get(option, 0.0)
    return (token_estimate / 1000.0) * cost_per_1k


def _is_viable(
    option: RoutingOption,
    policy: RoutingPolicy,
    specialists: set[str],
) -> tuple[bool, str]:
    """Check if a routing option is viable; return (viable, reason)."""
    if option == RoutingOption.LOCAL_SPECIALIST:
        if specialists:
            # If a SpecialistRegistry is installed, also require capability
            # match for the policy's task_type (defaults to "general").
            registry = _active_specialist_registry
            if registry is not None:
                task_type = getattr(policy, "task_type", "general") or "general"
                if not registry.can_handle(task_type):
                    return False, (
                        f"no registered specialist supports task_type={task_type!r}"
                    )
            return True, f"local specialist available: {next(iter(specialists))}"
        return False, "no local specialist deployed"

    if option == RoutingOption.LOCAL_GENERALIST:
        return True, "local generalist always available"

    if option in _EXTERNAL_API_OPTIONS:
        if not policy.allows_external_api():
            return False, (
                f"privacy tier {policy.privacy_tier.value!r} forbids external API"
            )
        if not _is_provider_alive(option):
            return False, f"provider {option.value} is currently unavailable"
        return True, f"external API {option.value} available"

    return False, f"unknown routing option: {option.value}"


def _is_privacy_violation(option: RoutingOption, policy: RoutingPolicy) -> bool:
    """True when the option requires an external API and policy forbids it."""
    return (
        option in _EXTERNAL_API_OPTIONS
        and policy.privacy_tier in (PrivacyTier.CONFIDENTIAL, PrivacyTier.SECRETS)
    )


def _api_endpoint(option: RoutingOption) -> str:
    """Return the API endpoint URL for an external API option."""
    _endpoints: dict[RoutingOption, str] = {
        RoutingOption.API_HAIKU: "https://api.anthropic.com/v1/messages",
        RoutingOption.API_SONNET: "https://api.anthropic.com/v1/messages",
        RoutingOption.API_DEEPSEEK: "https://api.deepseek.com/v1/chat/completions",
        RoutingOption.API_GPT5_MINI: "https://api.openai.com/v1/chat/completions",
    }
    return _endpoints.get(option, "")


def _build_receipt(task: str, option: RoutingOption, input_text: str) -> CostReceipt:
    """Build a stub CostReceipt for the dispatched route."""
    token_estimate = max(1, len(input_text.split()))
    cost_per_1k = _API_COST_PER_1K.get(option, 0.0)
    cost_usd = (token_estimate / 1000.0) * cost_per_1k

    return CostReceipt(
        task=task,
        route_taken=option.value,
        tokens_in=token_estimate,
        tokens_out=0,
        cost_usd=round(cost_usd, 8),
        latency_ms=0.0,
        local_displaced_estimate_usd=round(
            (token_estimate / 1000.0) * _API_COST_PER_1K.get(RoutingOption.API_HAIKU, 0.0),
            8,
        ),
        recorded_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Background health monitor (d36)
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Background thread that probes dead providers and auto-recovers them.

    Periodically iterates over _DEAD_PROVIDERS, performs a real HEAD probe
    via _health_check_with_timeout(), and removes recovered providers from
    the dead set. Independent of the TTL-based path in _is_provider_alive():
    HealthMonitor is for production deployments that want continuous probing,
    while the TTL path runs on-demand when dispatch() is called.

    Designed for safe daemon-thread usage in tests — start() and stop() are
    idempotent and stop() joins within the configured timeout.
    """

    def __init__(
        self,
        interval_seconds: float = 30.0,
        probe_timeout_seconds: float = 3.0,
        endpoint_resolver: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._interval = interval_seconds
        self._probe_timeout = probe_timeout_seconds
        self._endpoint_resolver = endpoint_resolver
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._iterations: int = 0
        self._recoveries: int = 0

    def _resolve_endpoint(self, provider_key: str) -> str:
        if self._endpoint_resolver is not None:
            return self._endpoint_resolver(provider_key)
        # Default: try to map back to a RoutingOption
        for opt in RoutingOption:
            if opt.value == provider_key:
                return _api_endpoint(opt) or ""
        return ""

    def probe_once(self) -> int:
        """Probe all currently dead providers exactly once.

        Returns the number of providers that recovered during this probe.
        """
        recovered = 0
        for key in list(_DEAD_PROVIDERS):
            endpoint = self._resolve_endpoint(key)
            if not endpoint:
                continue
            if _health_check_with_timeout(endpoint, timeout=self._probe_timeout):
                _DEAD_PROVIDERS.discard(key)
                _DEAD_PROVIDER_TIMESTAMPS.pop(key, None)
                _RECOVERY_ATTEMPTS.pop(key, None)
                _circuit_breaker[key] = 0
                recovered += 1
                EvidenceLogger.get_instance().emit(
                    {
                        "type": "route.health_monitor_recovery",
                        "data": {"provider": key, "probe_timeout_s": self._probe_timeout},
                    },
                    organ="route",
                )
        self._iterations += 1
        self._recoveries += recovered
        return recovered

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.probe_once()
            except Exception:
                pass  # never crash the monitor thread
            self._stop_event.wait(timeout=self._interval)

    def start(self) -> None:
        """Start the background probing thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="dojo-health-monitor")
        self._thread.start()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """Signal the thread to stop and join (idempotent)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def total_recoveries(self) -> int:
        return self._recoveries


__all__ = [
    "dispatch",
    "HealthMonitor",
    "_DEAD_PROVIDERS",
    "_HALF_OPEN_PROVIDERS",
    "_DEAD_PROVIDER_TIMESTAMPS",
    "_RECOVERY_TTL_SECONDS",
    "_circuit_breaker",
    "_latency_window",
    "_CIRCUIT_BREAKER_THRESHOLD",
    "_LATENCY_BREAKER_THRESHOLD_MS",
    "_LATENCY_BREAKER_WINDOW",
    "_record_provider_failure",
    "_record_provider_success",
    "_record_provider_latency",
    "_enter_half_open",
    "_is_half_open",
    "_health_check_with_timeout",
]
