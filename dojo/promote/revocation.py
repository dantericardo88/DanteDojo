"""Sovereign revocation check — HTTP GET to revocation_url at model load."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PassportRevoked
from dojo.core.one_door import one_door
from dojo.core.types import ConstitutionalPassport


@one_door("promote.revocation.check")
def check_revocation(
    passport: ConstitutionalPassport,
    timeout_seconds: float = 5.0,
) -> bool:
    """Check whether a passport has been revoked via HTTP GET to revocation_url.

    Emits REVOCATION_CHECKED evidence with the result.

    Args:
        passport: ConstitutionalPassport with revocation_url field.
        timeout_seconds: HTTP timeout for the revocation check.

    Returns:
        False (not revoked) — never returns True; raises PassportRevoked instead.

    Raises:
        PassportRevoked: if the revocation endpoint returns {"revoked": true}.
    """
    logger = EvidenceLogger.get_instance()
    revocation_url = passport.revocation_url
    passport_id = passport.passport_id

    logger.emit(
        {
            "type": EvidenceType.REVOCATION_CHECKED.value,
            "data": {
                "passport_id": passport_id,
                "revocation_url": revocation_url,
                "status": "checking",
            },
        },
        organ="promote",
    )

    try:
        req = urllib.request.Request(
            f"{revocation_url}?passport_id={passport_id}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            revoked = bool(data.get("revoked", False))
    except urllib.error.URLError:
        # Network failure: fail-open (allow) but log
        logger.emit(
            {
                "type": EvidenceType.REVOCATION_CHECKED.value,
                "data": {
                    "passport_id": passport_id,
                    "status": "network_failure_allow",
                    "reason": "revocation endpoint unreachable",
                },
            },
            organ="promote",
        )
        return False
    except (json.JSONDecodeError, KeyError):
        # Malformed response: fail-open
        logger.emit(
            {
                "type": EvidenceType.REVOCATION_CHECKED.value,
                "data": {
                    "passport_id": passport_id,
                    "status": "parse_error_allow",
                },
            },
            organ="promote",
        )
        return False

    logger.emit(
        {
            "type": EvidenceType.REVOCATION_CHECKED.value,
            "data": {
                "passport_id": passport_id,
                "revocation_url": revocation_url,
                "status": "revoked" if revoked else "valid",
                "revoked": revoked,
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


__all__ = ["check_revocation"]
