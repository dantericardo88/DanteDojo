"""d45 integration tests — real wandb in offline mode.

Uses wandb.init(mode='offline') so no internet or W&B account is required.
All writes land in a local temp directory. This IS a real W&B test, not a mock.

Verifies:
- WandBAdapter.init_run() creates a real wandb run when wandb is available
- log() / log_config() / finish() work against the real wandb SDK
- WandBTrainingCallback.on_log() fires and logs metrics
- EXPERIMENT_RUN_STARTED evidence emitted on init_run()
- All paths pass without DOJO_REAL_GPU=1
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import pytest

wandb = pytest.importorskip("wandb", reason="wandb not installed")

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.tracker import RunConfig
from dojo.experiment.wandb_adapter import WandBAdapter, WandBTrainingCallback


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture()
def wandb_tmp_dir(tmp_path):
    """Provide a clean temp dir for offline wandb artifacts."""
    d = tmp_path / "wandb_offline"
    d.mkdir()
    return str(d)


@pytest.fixture()
def offline_adapter(wandb_tmp_dir):
    """WandBAdapter wired to offline wandb pointing at a temp dir."""
    os.environ["WANDB_DIR"] = wandb_tmp_dir
    os.environ["WANDB_MODE"] = "offline"
    adapter = WandBAdapter(project="dojo-test", use_wandb=True)
    yield adapter
    try:
        if adapter._run is not None:
            adapter.finish()
    except Exception:
        pass
    os.environ.pop("WANDB_MODE", None)
    os.environ.pop("WANDB_DIR", None)


# ---------------------------------------------------------------------------
# WandBAdapter offline integration tests
# ---------------------------------------------------------------------------

class TestWandBAdapterOffline:

    def test_init_run_creates_real_wandb_run(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-001")
        run_id = offline_adapter.init_run(config)
        assert run_id
        assert wandb.run is not None

    def test_wandb_run_is_not_none_after_init(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-002")
        offline_adapter.init_run(config)
        assert offline_adapter._run is not None

    def test_wandb_available_flag_set(self, offline_adapter):
        assert offline_adapter._wandb_available is True

    def test_log_metrics_does_not_raise(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-003")
        offline_adapter.init_run(config)
        offline_adapter.log({"loss": 0.45, "accuracy": 0.82}, step=1)

    def test_log_emits_experiment_metric_evidence(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-004")
        offline_adapter.init_run(config)
        offline_adapter.log({"loss": 0.31}, step=2)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_METRIC.value in types

    def test_init_run_emits_run_started_evidence(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-005")
        offline_adapter.init_run(config)
        events = EvidenceLogger.get_events()
        started = [e for e in events if e.get("type") == EvidenceType.EXPERIMENT_RUN_STARTED.value]
        assert len(started) >= 1
        assert started[0]["data"]["wandb_available"] is True

    def test_log_config_does_not_raise(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-006")
        offline_adapter.init_run(config)
        offline_adapter.log_config({"lr": 2e-4, "epochs": 3})

    def test_finish_emits_run_complete_evidence(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-007")
        offline_adapter.init_run(config)
        offline_adapter.finish()
        events = EvidenceLogger.get_events()
        complete = [e for e in events if e.get("type") == EvidenceType.EXPERIMENT_RUN_COMPLETE.value]
        assert len(complete) >= 1

    def test_finish_clears_run(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-008")
        offline_adapter.init_run(config)
        offline_adapter.finish()
        assert offline_adapter._run is None
        assert offline_adapter._active_run_id is None

    def test_multiple_log_calls_all_emit_evidence(self, offline_adapter):
        config = RunConfig(experiment_name="offline-exp", run_name="run-offline-009")
        offline_adapter.init_run(config)
        for step in range(3):
            offline_adapter.log({"loss": 1.0 - step * 0.1}, step=step)
        events = EvidenceLogger.get_events()
        metric_events = [e for e in events if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
                         and "metrics" in e.get("data", {})]
        assert len(metric_events) >= 3


# ---------------------------------------------------------------------------
# WandBTrainingCallback offline integration tests
# ---------------------------------------------------------------------------

class TestWandBTrainingCallbackOffline:

    @pytest.fixture()
    def adapter_cb(self, offline_adapter):
        config = RunConfig(experiment_name="cb-exp", run_name="cb-run-001")
        offline_adapter.init_run(config)
        cb = offline_adapter.as_callback()
        return offline_adapter, cb

    def test_on_step_fires_and_logs(self, adapter_cb):
        adapter, cb = adapter_cb
        cb.on_step(step=0, metrics={"loss": 0.9, "lr": 1e-4})
        events = EvidenceLogger.get_events()
        metric_events = [e for e in events
                         if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
                         and e.get("data", {}).get("step") == 0]
        assert len(metric_events) >= 1

    def test_on_step_metric_keys_in_evidence(self, adapter_cb):
        _, cb = adapter_cb
        cb.on_step(step=1, metrics={"loss": 0.85, "grad_norm": 0.9})
        events = EvidenceLogger.get_events()
        log_events = [e for e in events
                      if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
                      and "metrics" in e.get("data", {})]
        last = log_events[-1]["data"]["metrics"]
        assert "loss" in last

    def test_on_epoch_end_prefixes_keys(self, adapter_cb):
        _, cb = adapter_cb
        cb.on_epoch_end(epoch=1, metrics={"loss": 0.3, "f1": 0.77})
        events = EvidenceLogger.get_events()
        log_events = [e for e in events
                      if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
                      and "metrics" in e.get("data", {})]
        last_metrics = log_events[-1]["data"]["metrics"]
        assert "epoch_loss" in last_metrics
        assert "epoch_f1" in last_metrics

    def test_on_train_end_finishes_run(self, adapter_cb):
        adapter, cb = adapter_cb
        cb.on_train_end(final_metrics={"final_loss": 0.1})
        assert adapter._active_run_id is None

    def test_on_train_end_emits_run_complete_evidence(self, adapter_cb):
        _, cb = adapter_cb
        cb.on_train_end(final_metrics={"accuracy": 0.95})
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EXPERIMENT_RUN_COMPLETE.value in types

    def test_callback_on_log_style_step_sequence(self, adapter_cb):
        """Simulate a training loop calling on_step at each step."""
        _, cb = adapter_cb
        losses = [0.9, 0.7, 0.5, 0.3]
        for step, loss in enumerate(losses):
            cb.on_step(step=step, metrics={"loss": loss})
        events = EvidenceLogger.get_events()
        step_metric_events = [e for e in events
                               if e.get("type") == EvidenceType.EXPERIMENT_METRIC.value
                               and "step" in e.get("data", {})]
        assert len(step_metric_events) >= len(losses)
