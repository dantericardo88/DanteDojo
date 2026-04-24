"""dojo/precision/fp8.py — FP8 precision handler with hard Ampere rejection.

FP8 (E4M3/E5M2) compute requires SM >= 8.9 (Ada Lovelace, Hopper, Blackwell).
Ampere (SM 8.6, e.g. RTX 3090) does not support FP8 natively — attempting to
use it there raises PrecisionUnsupported with a clear remediation hint.
"""

from __future__ import annotations

from dojo.core.exceptions import PrecisionUnsupported
from dojo.core.types import HALProfile


class FP8Handler:
    """Validate and gate FP8 precision usage against hardware capabilities.

    All methods are static — no instance state is required.
    """

    # Minimum compute capability (SM version) for native FP8 support.
    # Ada Lovelace: 8.9, Hopper: 9.0, Blackwell: 10.0+
    _MIN_CC_FP8 = 8.9

    @staticmethod
    def ensure_supported(profile: HALProfile) -> None:
        """Assert that FP8 is natively supported on *profile*'s hardware.

        Raises PrecisionUnsupported (not a warning — hard fail) if the
        compute capability is below SM 8.9.  Ampere (SM 8.6) is explicitly
        rejected with a routing hint.

        Args:
            profile: The HALProfile to check.

        Raises:
            PrecisionUnsupported: If ``profile.compute_capability`` < 8.9.
        """
        try:
            cc = float(profile.compute_capability)
        except (ValueError, TypeError) as exc:
            raise PrecisionUnsupported(
                f"Cannot parse compute_capability '{profile.compute_capability}' as float",
                reason="Ensure HALProfile.compute_capability is 'major.minor' string (e.g. '8.6').",
            ) from exc

        if cc < FP8Handler._MIN_CC_FP8:
            raise PrecisionUnsupported(
                f"FP8 requires SM >= 8.9; got SM {profile.compute_capability} ({profile.gpu_name})",
                reason=(
                    "Route to BF16 or W8A16-Marlin for Ampere hardware.  "
                    "Use dojo.turbo.policy.get_policies_for_profile() to obtain "
                    "the correct policy list for this HALProfile."
                ),
            )

    @staticmethod
    def is_fp8_supported(profile: HALProfile) -> bool:
        """Return True if FP8 is natively supported on *profile*'s hardware.

        This is the non-raising companion to ``ensure_supported``.

        Args:
            profile: The HALProfile to check.

        Returns:
            True if SM >= 8.9, False otherwise.
        """
        try:
            return float(profile.compute_capability) >= FP8Handler._MIN_CC_FP8
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Module-level convenience alias
# ---------------------------------------------------------------------------

def is_fp8_supported(profile: HALProfile) -> bool:
    """Module-level shorthand for FP8Handler.is_fp8_supported."""
    return FP8Handler.is_fp8_supported(profile)


__all__ = ["FP8Handler", "is_fp8_supported"]
