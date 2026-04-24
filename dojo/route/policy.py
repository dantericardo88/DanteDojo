"""Routing policy helpers for DanteDojo.

Provides default RoutingPolicy objects for common Dante task names
and the TASK_DEFAULTS preference order table.
"""

from __future__ import annotations

from dojo.core.types import PrivacyTier, RoutingOption, RoutingPolicy


# ---------------------------------------------------------------------------
# Default preference order per task
# ---------------------------------------------------------------------------

TASK_DEFAULTS: dict[str, list[RoutingOption]] = {
    "bookkeeping": [RoutingOption.LOCAL_SPECIALIST, RoutingOption.LOCAL_GENERALIST],
    "b2b_email": [RoutingOption.LOCAL_SPECIALIST, RoutingOption.API_HAIKU],
    "code_review": [RoutingOption.LOCAL_SPECIALIST, RoutingOption.API_SONNET],
    "ocr_cleanup": [RoutingOption.LOCAL_SPECIALIST, RoutingOption.API_HAIKU],
    "sop_extraction": [RoutingOption.LOCAL_SPECIALIST, RoutingOption.API_HAIKU],
    "general": [RoutingOption.API_HAIKU],
}

# Defaults for policies not in TASK_DEFAULTS
_DEFAULT_MAX_LATENCY_MS = 30_000
_DEFAULT_MAX_COST_USD = 0.01
_DEFAULT_FALLBACK = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def default_policy_for_task(
    task_name: str,
    privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
) -> RoutingPolicy:
    """Return a sensible default RoutingPolicy for a known Dante task name.

    Falls back to the "general" task policy if task_name is unknown.

    Args:
        task_name: Dante task identifier (e.g. "bookkeeping", "b2b_email").
        privacy_tier: Privacy level that governs external API access.

    Returns:
        RoutingPolicy with sane defaults for the task.
    """
    options = TASK_DEFAULTS.get(task_name, TASK_DEFAULTS["general"])

    return RoutingPolicy(
        task_name=task_name,
        preference_order=tuple(options),
        max_latency_ms=_DEFAULT_MAX_LATENCY_MS,
        max_cost_usd=_DEFAULT_MAX_COST_USD,
        privacy_tier=privacy_tier,
        fallback_on_unavailable=_DEFAULT_FALLBACK,
    )


__all__ = ["TASK_DEFAULTS", "default_policy_for_task"]
