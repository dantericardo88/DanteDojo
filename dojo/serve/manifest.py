"""Deployment manifest builder for DanteDojo.

Builds and hashes a DeploymentManifest so every server instance is
cryptographically identified. The manifest_sha256 is SHA-256 of to_dict().
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from dojo.core.types import Backend, DeploymentManifest


def build_manifest(
    model_artifact_id: str,
    backend: Backend,
    endpoint_url: str,
    context_length: int,
    tp_size: int = 1,
    dtype: str = "bfloat16",
    passport_id: str = "",
    pid: int = 0,
    quantization: Optional[str] = None,
) -> DeploymentManifest:
    """Build and hash a DeploymentManifest.

    The manifest_sha256 field is the SHA-256 hex digest of the JSON
    representation of all fields (excluding manifest_sha256 itself),
    with keys sorted for determinism.

    Args:
        model_artifact_id: Dojo artifact URI (e.g. "dojo://qwen-7b/v1").
        backend: Inference backend enum value.
        endpoint_url: HTTP endpoint where the model is served.
        context_length: Maximum context window in tokens.
        tp_size: Tensor-parallel degree (1 = single GPU).
        dtype: Floating-point dtype string ("bfloat16", "float16", etc.).
        passport_id: Constitutional passport identifier (empty = no passport).
        pid: OS process ID of the server subprocess (0 = not started yet).
        quantization: Optional quantization tag (e.g. "Q4_K_M", "gptq_int4").

    Returns:
        Frozen DeploymentManifest with a populated manifest_sha256.
    """
    started_at = datetime.now(timezone.utc)

    # Build the hashable payload (everything except manifest_sha256)
    hashable: dict = {
        "model_artifact_id": model_artifact_id,
        "backend": backend.value,
        "endpoint_url": endpoint_url,
        "context_length": context_length,
        "quantization": quantization,
        "tp_size": tp_size,
        "dtype": dtype,
        "passport_id": passport_id,
        "started_at": started_at.isoformat(),
        "pid": pid,
    }

    manifest_sha256 = hashlib.sha256(
        json.dumps(hashable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return DeploymentManifest(
        model_artifact_id=model_artifact_id,
        backend=backend,
        endpoint_url=endpoint_url,
        context_length=context_length,
        quantization=quantization,
        tp_size=tp_size,
        dtype=dtype,
        passport_id=passport_id,
        started_at=started_at,
        pid=pid,
        manifest_sha256=manifest_sha256,
    )


__all__ = ["build_manifest"]
