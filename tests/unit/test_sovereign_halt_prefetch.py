"""Tests for SovereignHalt firing when lookahead > 16 is attempted."""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError, SovereignHalt
from dojo.stream.prefetch_governor import PrefetchGovernor, PrefetchGovernorError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_governor(max_lookahead: int = 16) -> PrefetchGovernor:
    """Return a mock-mode governor for CI-safe tests."""
    EvidenceLogger.clear()
    return PrefetchGovernor(max_lookahead=max_lookahead, mock_mode=True)


# ---------------------------------------------------------------------------
# 1. SovereignHalt fires when 17 layers issued in one batch
# ---------------------------------------------------------------------------

def test_sovereign_halt_fires_at_17_layers() -> None:
    gov = make_governor(max_lookahead=16)
    layers = [f"layer_{i}" for i in range(17)]
    with pytest.raises(SovereignHalt):
        gov.issue_prefetch(layers)


# ---------------------------------------------------------------------------
# 2. SovereignHalt fires at exactly 17 (16 pending + 1 new)
# ---------------------------------------------------------------------------

def test_sovereign_halt_fires_at_exactly_17() -> None:
    gov = make_governor(max_lookahead=16)
    # Fill pending to 16 by issuing in groups that fit
    gov.issue_prefetch([f"layer_{i}" for i in range(16)])
    assert len(gov.pending) == 16

    with pytest.raises(SovereignHalt):
        gov.issue_prefetch(["layer_extra"])


# ---------------------------------------------------------------------------
# 3. No exception at exactly 16 layers
# ---------------------------------------------------------------------------

def test_sovereign_halt_passes_at_16() -> None:
    gov = make_governor(max_lookahead=16)
    layers = [f"layer_{i}" for i in range(16)]
    result = gov.issue_prefetch(layers)
    assert result.status == "PrefetchIssued"
    assert len(result.layers_issued) == 16


# ---------------------------------------------------------------------------
# 4. SovereignHalt is a subclass of DojoError
# ---------------------------------------------------------------------------

def test_sovereign_halt_is_dojo_error_subclass() -> None:
    exc = SovereignHalt("test message", reason="test reason")
    assert isinstance(exc, DojoError)
    assert isinstance(exc, SovereignHalt)


# ---------------------------------------------------------------------------
# 5. Evidence event is emitted when halt fires
# ---------------------------------------------------------------------------

def test_sovereign_halt_evidence_emitted_on_breach() -> None:
    EvidenceLogger.clear()
    gov = make_governor(max_lookahead=16)

    layers = [f"layer_{i}" for i in range(17)]
    with pytest.raises(SovereignHalt):
        gov.issue_prefetch(layers)

    events = EvidenceLogger.get_events()
    halt_events = [
        e for e in events
        if e.get("type") == EvidenceType.SOVEREIGN_HALT.value
    ]
    assert len(halt_events) >= 1, "Expected at least one sovereign_halt evidence event"
    data = halt_events[0].get("data", {})
    assert data.get("event") == "sovereign_halt"
    assert data.get("lookahead_requested") == 17
    assert data.get("max_lookahead") == 16


# ---------------------------------------------------------------------------
# 6. __init__ rejects max_lookahead > 16
# ---------------------------------------------------------------------------

def test_prefetch_governor_init_rejects_above_16() -> None:
    with pytest.raises(PrefetchGovernorError):
        PrefetchGovernor(max_lookahead=17, mock_mode=True)


# ---------------------------------------------------------------------------
# 7. __init__ accepts max_lookahead == 16
# ---------------------------------------------------------------------------

def test_prefetch_governor_init_accepts_16() -> None:
    gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
    assert gov.max_lookahead == 16


# ---------------------------------------------------------------------------
# 8. simulate_512_token_inference — no halts when pending is cleared each step
# ---------------------------------------------------------------------------

def test_simulate_512_inference_no_halts() -> None:
    gov = make_governor(max_lookahead=16)
    # layers_per_step=3 is well within MAX_LOOKAHEAD=16; pending cleared each step
    stats = gov.simulate_512_token_inference(layers_per_step=3, n_steps=8)
    assert stats["steps"] == 8
    assert stats["prefetches_issued"] == 24  # 3 * 8
    assert stats["halts_triggered"] == 0


# ---------------------------------------------------------------------------
# 9. simulate_512_token_inference — halts fire when pending is NOT cleared
#    We monkey-patch the method so pending accumulates across steps.
# ---------------------------------------------------------------------------

def test_simulate_512_inference_halts_when_not_cleared() -> None:
    gov = make_governor(max_lookahead=16)

    halts = 0
    total_prefetches = 0
    # Do NOT clear pending between steps — pending accumulates
    for step in range(8):
        layers = [f"layer_{step * 3 + i}" for i in range(3)]
        try:
            gov.issue_prefetch(layers)
            total_prefetches += len(layers)
            # Intentionally NOT clearing pending
        except SovereignHalt:
            halts += 1

    # After 5 steps: 15 pending. Step 6 adds 3 → 18 > 16 → halt.
    # Steps 6, 7, 8 all halt (pending stays at 15 after first halt).
    assert halts > 0, "Expected at least one SovereignHalt when pending accumulates"


# ---------------------------------------------------------------------------
# 10. Exception message mentions "lookahead" and "max"
# ---------------------------------------------------------------------------

def test_prefetch_governor_halt_message_informative() -> None:
    gov = make_governor(max_lookahead=16)
    layers = [f"layer_{i}" for i in range(17)]
    with pytest.raises(SovereignHalt) as exc_info:
        gov.issue_prefetch(layers)

    message = str(exc_info.value).lower()
    assert "lookahead" in message, f"Expected 'lookahead' in: {message}"
    assert "max" in message or "16" in message, f"Expected 'max' or '16' in: {message}"
