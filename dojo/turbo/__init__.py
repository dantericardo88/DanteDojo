"""dojo/turbo — Autonomous GPU optimizer (turbocharge command)."""

from dojo.turbo.config import ProfileResult, TurboConfig, TurboResult
from dojo.turbo.optimizer import select_optimal
from dojo.turbo.policy import PolicyType, get_policies_for_profile

# Import as run_turbocharge to avoid shadowing the dojo.turbo.turbocharge MODULE
# with the turbocharge() function — shadowing breaks monkeypatch in tests.
from dojo.turbo.turbocharge import turbocharge as run_turbocharge

__all__ = [
    "TurboConfig",
    "TurboResult",
    "ProfileResult",
    "PolicyType",
    "get_policies_for_profile",
    "select_optimal",
    "run_turbocharge",
]
