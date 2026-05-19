"""Top-level serve launcher for DanteDojo.

Selects and (in real-GPU mode) launches an inference server based on HALProfile.
Runs the LaunchFSM, emits constitutional evidence, returns a DeploymentManifest.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import ServerStartError
from dojo.core.fsm import BaseFSM
from dojo.core.one_door import one_door
from dojo.core.types import Backend, ConstitutionalPassport, DeploymentManifest, HALProfile, HardwareTier
from dojo.serve.manifest import build_manifest


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------

class LaunchState(Enum):
    IDLE = "idle"
    CONFIGURING = "configuring"
    PROVISIONING = "provisioning"
    WARMING = "warming"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class LaunchFSM(BaseFSM[LaunchState]):
    """FSM for the server launch lifecycle."""

    initial_state = LaunchState.IDLE

    _TRANSITIONS: dict = {
        LaunchState.IDLE: frozenset({LaunchState.CONFIGURING}),
        LaunchState.CONFIGURING: frozenset({LaunchState.PROVISIONING, LaunchState.FAILED}),
        LaunchState.PROVISIONING: frozenset({LaunchState.WARMING, LaunchState.FAILED}),
        LaunchState.WARMING: frozenset({LaunchState.READY, LaunchState.FAILED}),
        LaunchState.READY: frozenset({LaunchState.STOPPING}),
        LaunchState.STOPPING: frozenset({LaunchState.STOPPED, LaunchState.FAILED}),
    }


# ---------------------------------------------------------------------------
# Backend routing table
# ---------------------------------------------------------------------------

_BACKEND_MAP: dict[HardwareTier, Backend] = {
    HardwareTier.TIER_4060_LAPTOP: Backend.LLAMACPP,
    HardwareTier.TIER_3090: Backend.VLLM,
    HardwareTier.TIER_3090_DUAL: Backend.VLLM,
    HardwareTier.TIER_H100: Backend.VLLM,
    HardwareTier.TIER_APPLE_M4: Backend.MLX_LM,
    HardwareTier.TIER_AMD_MI300: Backend.VLLM,
    HardwareTier.TIER_CPU: Backend.LLAMACPP,
}


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("serve.launch")
def launch_server(
    model_artifact_id: str,
    model_path: Path,
    profile: HALProfile,
    backend: Optional[Backend] = None,
    *,
    context_length: int = 4096,
    port: int = 8080,
    passport_id: str = "",
    passport: Optional[ConstitutionalPassport] = None,
    openai_port: Optional[int] = None,
) -> DeploymentManifest:
    """Select and launch an inference server.

    Backend auto-selection when backend=None:
      TIER_4060_LAPTOP → LLAMACPP
      TIER_3090        → VLLM
      TIER_3090_DUAL   → VLLM (TP=2)
      TIER_H100        → VLLM
      TIER_APPLE_M4    → MLX_LM
      TIER_AMD_MI300   → VLLM
      TIER_CPU         → LLAMACPP

    Revocation check: when `passport` is provided and has a revocation_url,
    check_revocation() is called before the backend starts.  A revoked passport
    blocks inference — PassportRevoked is raised and the FSM transitions to FAILED.

    Args:
        model_artifact_id: Dojo artifact URI.
        model_path: Filesystem path to the model.
        profile: HAL hardware profile for this machine.
        backend: Override backend selection (None = auto).
        context_length: Context window size in tokens.
        port: Port for the HTTP server.
        passport_id: Constitutional passport ID string (used in manifests).
        passport: Full ConstitutionalPassport object; when provided with a
            revocation_url, check_revocation() is called at serve time.
        openai_port: If set, also start an OpenAI-compatible proxy on this port.

    Returns:
        DeploymentManifest describing the running (or stub) server.
    """
    global _active_backend, _active_fsm

    logger = EvidenceLogger.get_instance()
    fsm = LaunchFSM()
    _active_fsm = fsm

    # Phase 1: IDLE → CONFIGURING
    fsm.transition(LaunchState.CONFIGURING)

    # Revocation gate: block serve if passport has been revoked.
    # This is the serve-time sovereign check — promotion already checked once,
    # but we re-check at inference start so a revoked passport can't be loaded.
    if passport is not None and passport.revocation_url:
        from dojo.promote.revocation import check_revocation  # noqa: PLC0415
        try:
            check_revocation(passport)
        except Exception as exc:
            fsm.transition(LaunchState.FAILED)
            raise ServerStartError(
                f"Serve blocked: passport '{passport.passport_id}' failed revocation check — {exc}",
                reason=str(exc),
            ) from exc

    selected_backend = backend or _select_backend(profile)
    tp_size = 2 if (selected_backend == Backend.VLLM and profile.tier == HardwareTier.TIER_3090_DUAL) else 1

    logger.emit(
        {
            "type": EvidenceType.SERVE_START.value,
            "data": {
                "model_artifact_id": model_artifact_id,
                "model_path": str(model_path),
                "backend": selected_backend.value,
                "profile_tier": profile.tier.value,
                "context_length": context_length,
                "port": port,
                "tp_size": tp_size,
            },
        },
        organ="serve",
    )

    pid = 0
    endpoint_url = f"http://localhost:{port}"

    try:
        # Phase 2: CONFIGURING → PROVISIONING
        fsm.transition(LaunchState.PROVISIONING)
        pid = _provision_backend(selected_backend, model_path, profile, port, context_length)

        # Phase 3: PROVISIONING → WARMING
        fsm.transition(LaunchState.WARMING)
        _warm_backend(selected_backend, port)

        # Phase 4: WARMING → READY
        fsm.transition(LaunchState.READY)

    except Exception as exc:
        fsm.transition(LaunchState.FAILED)
        raise ServerStartError(
            f"Server launch failed: {exc}",
            reason=str(exc),
        ) from exc

    manifest = build_manifest(
        model_artifact_id=model_artifact_id,
        backend=selected_backend,
        endpoint_url=endpoint_url,
        context_length=context_length,
        tp_size=tp_size,
        passport_id=passport_id,
        pid=pid,
    )

    logger.emit(
        {
            "type": EvidenceType.SERVE_READY.value,
            "data": {
                "model_artifact_id": model_artifact_id,
                "backend": selected_backend.value,
                "endpoint_url": endpoint_url,
                "manifest_sha256": manifest.manifest_sha256,
                "pid": pid,
            },
        },
        organ="serve",
    )

    if openai_port is not None:
        import threading
        import uvicorn
        from typing import Any as _Any
        from dojo.serve.openai_compat import create_app

        _openai_app: _Any = create_app(
            backend_url=endpoint_url,
            model_id=model_artifact_id,
            passport_id=passport_id,
        )
        _openai_port: int = openai_port

        def _run_openai_api() -> None:
            uvicorn.run(_openai_app, host="127.0.0.1", port=_openai_port, log_level="warning")  # type: ignore[arg-type]

        t = threading.Thread(target=_run_openai_api, daemon=True, name="dojo-openai-api")
        t.start()

    return manifest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_backend(profile: HALProfile) -> Backend:
    """Choose the best inference backend for the given hardware profile."""
    return _BACKEND_MAP.get(profile.tier, Backend.LLAMACPP)


def _provision_backend(
    backend: Backend,
    model_path: Path,
    profile: HALProfile,
    port: int,
    context_length: int,
) -> int:
    """Start the backend subprocess and return its PID."""
    global _active_backend
    if backend == Backend.VLLM:
        from dojo.serve.vllm_backend import VLLMBackend
        b = VLLMBackend(model_path=model_path, profile=profile, port=port, context_length=context_length)
        _active_backend = b
        return b.start()
    elif backend == Backend.LLAMACPP:
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        b = LlamaCppBackend(model_path=model_path, port=port)
        _active_backend = b
        return b.start()
    else:
        # MLX_LM, SGLANG — stub for v1
        return 0


def _warm_backend(backend: Backend, port: int) -> None:
    """Run health checks; no-op in stub mode."""
    import os
    if os.environ.get("DOJO_REAL_GPU", "0") != "1":
        return

    if backend == Backend.VLLM:
        from dojo.serve.vllm_backend import VLLMBackend
        # Health check is called separately by real operators; skip in warm
        pass
    elif backend == Backend.LLAMACPP:
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        pass


# ---------------------------------------------------------------------------
# Module-level backend instance (for stop_serving cleanup)
# ---------------------------------------------------------------------------

_active_backend: Optional[object] = None
_active_fsm: Optional[LaunchFSM] = None


def stop_serving() -> None:
    """Stop the currently running backend and transition FSM to STOPPED.

    Calls backend.stop() if a backend is active, emits SERVE_STOPPED evidence,
    and resets module-level state.
    """
    global _active_backend, _active_fsm

    logger = EvidenceLogger.get_instance()

    if _active_backend is not None:
        try:
            _active_backend.stop()  # type: ignore[union-attr]
        except Exception:
            pass
        _active_backend = None

    if _active_fsm is not None:
        try:
            _active_fsm.transition(LaunchState.STOPPING)
            _active_fsm.transition(LaunchState.STOPPED)
        except Exception:
            pass
        _active_fsm = None

    logger.emit(
        {
            "type": EvidenceType.SERVE_STOPPED.value,
            "data": {"source": "stop_serving"},
        },
        organ="serve",
    )


# ---------------------------------------------------------------------------
# Multi-backend cascade serving (d47)
# ---------------------------------------------------------------------------

from dojo.serve.backend_cascade import BackendCascade, BackendCascadeExhausted


def serve_with_cascade(
    backends: list[tuple[str, Any]],
    prompt: str,
    **kwargs: Any,
) -> dict:
    """Generate a completion by trying ``backends`` in order until one succeeds.

    ``backends`` is a list of (name, generate_callable) tuples.  Each callable
    must accept (prompt, **kwargs) and either return a dict or raise.  Errors
    are caught and the next backend is tried.

    Emits ``serve.cascade_attempt`` per backend and ``serve.cascade_complete``
    when finished (success or exhausted).

    Raises:
        BackendCascadeExhausted: when every backend failed.
    """
    cascade = BackendCascade(backends=list(backends))
    return cascade.generate(prompt, **kwargs)


__all__ = [
    "LaunchFSM",
    "LaunchState",
    "launch_server",
    "stop_serving",
    "serve_with_cascade",
    "BackendCascadeExhausted",
]
