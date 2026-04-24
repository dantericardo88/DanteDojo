"""DanteDojo promotion gate — DanteForge policy check.

POSTs the candidate manifest to DanteForge /v1/policy/verify.
Fail-closed: timeout or any error → BLOCKED, never GREEN.
Circuit breaker: 3 retries with exponential backoff.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from dojo.core.evidence import EvidenceLogger, EvidenceType

if TYPE_CHECKING:
    from dojo.core.types import ModelArtifact
    from dojo.promote.gate import GateCheckResult


def _make_receipt(gate: str, passed: bool, reasons: list[str]) -> bytes:
    payload = json.dumps(
        {"gate": gate, "passed": passed, "reasons": reasons, "ts": time.time()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).digest()


def _mock_forge_verify(candidate: "ModelArtifact") -> dict[str, Any]:
    """Default mock — always returns GREEN."""
    return {
        "passed": True,
        "reasons": [],
        "receipt": hashlib.sha256(candidate.artifact_id.encode()).hexdigest(),
    }


def check_forge(
    candidate: "ModelArtifact",
    *,
    forge_client: Any = None,
    timeout_seconds: float = 60.0,
) -> "GateCheckResult":
    """HTTP POST to DanteForge /v1/policy/verify.

    If forge_client is None → use mock that returns GREEN.
    Timeout: 60s. Fail-closed: timeout or error → BLOCKED, never GREEN.
    Circuit breaker: 3 retries with exponential backoff.

    Returns GateCheckResult. Emits PROMOTE_FORGE_GREEN or PROMOTE_FORGE_BLOCKED.
    """
    from dojo.promote.gate import GateCheckResult  # noqa: PLC0415

    logger = EvidenceLogger.get_instance()
    reasons: list[str] = []
    gate = "forge_check"
    max_retries = 3
    backoff_base = 1.0  # seconds

    try:
        if forge_client is None:
            # Mock path — always GREEN
            response = _mock_forge_verify(candidate)
        else:
            # Real client with retries + exponential backoff
            response = None
            last_exc: Exception | None = None

            for attempt in range(max_retries):
                try:
                    response = forge_client.verify_policy(
                        candidate.to_dict(),
                        timeout=timeout_seconds,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        sleep_time = backoff_base * (2 ** attempt)
                        time.sleep(sleep_time)

            if response is None:
                reasons.append(
                    f"Forge policy verification failed after {max_retries} attempts: {last_exc}"
                )

        if response is not None:
            forge_passed = bool(response.get("passed", False))
            forge_reasons: list[str] = list(response.get("reasons", []))
            if not forge_passed:
                reasons.extend(forge_reasons if forge_reasons else ["Forge policy check failed"])

    except Exception as exc:
        # Fail-closed: any unexpected error → BLOCKED
        reasons.append(f"Unexpected error in forge_check: {exc}")

    passed = len(reasons) == 0
    receipt = _make_receipt(gate, passed, reasons)

    result = GateCheckResult(gate=gate, passed=passed, reasons=reasons, receipt=receipt)

    ev_type = EvidenceType.PROMOTE_FORGE_GREEN if passed else EvidenceType.PROMOTE_FORGE_BLOCKED
    logger.emit(
        {
            "type": ev_type.value,
            "data": {
                "gate": gate,
                "passed": passed,
                "artifact_id": candidate.artifact_id,
                "reasons": reasons,
                "timeout_seconds": timeout_seconds,
            },
        },
        organ="promote",
    )

    return result
