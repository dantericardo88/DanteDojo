"""dojo/hal/profile.py — HAL profile signing and verification.

Provides a deterministic SHA-256 signature for a HALProfile.  The hash is
computed over the profile's stable JSON with ``profile_sha256`` excluded from
the payload — this gives a proper fixed-point so that:

    sha = sign_profile(profile)        # excludes profile_sha256 field
    profile2 = replace(profile, profile_sha256=sha)
    verify_profile_signature(profile2) is True

No external crypto dependency is required (stdlib hashlib only).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from dojo.core.types import HALProfile


def _signable_dict(profile: HALProfile) -> Dict[str, Any]:
    """Return the profile's to_dict() with profile_sha256 removed.

    Excluding the hash field from the signed payload avoids the
    chicken-and-egg problem and allows straightforward verification.
    """
    d = profile.to_dict()
    d.pop("profile_sha256", None)
    return d


def sign_profile(profile: HALProfile) -> str:
    """Return a SHA-256 hex digest that signs the profile's content fields.

    The ``profile_sha256`` field itself is excluded from the payload so that
    the signature can be stored in the profile without creating a circular
    dependency.  All other fields from ``profile.to_dict()`` are included,
    serialised with ``sort_keys=True`` for determinism.

    Args:
        profile: The HALProfile to sign.

    Returns:
        A 64-character lowercase hex string (SHA-256 digest).
    """
    payload = json.dumps(
        _signable_dict(profile),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_profile_signature(profile: HALProfile) -> bool:
    """Return True if the profile's stored ``profile_sha256`` matches a
    freshly computed signature of the profile's content fields.

    Args:
        profile: The HALProfile to verify.

    Returns:
        True if ``profile.profile_sha256 == sign_profile(profile)``; False
        otherwise.
    """
    expected = sign_profile(profile)
    return profile.profile_sha256 == expected


__all__ = ["sign_profile", "verify_profile_signature"]
