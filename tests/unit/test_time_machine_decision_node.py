import json
from pathlib import Path

from dojo.core.evidence import EvidenceLogger
from dojo.core.time_machine_decision_node import record_decision_node, resolve_decision_store


def test_records_product_local_decision_node(tmp_path: Path) -> None:
    node = record_decision_node(
        project_root=tmp_path,
        session_id="train-a",
        actor_id="trainer",
        prompt="train.finetune.started",
        result={"base_model": "tiny"},
        success=True,
    )

    store = resolve_decision_store(tmp_path)
    parsed = json.loads(store.read_text(encoding="utf-8").strip())

    assert parsed["id"] == node["id"]
    assert parsed["actor"]["product"] == "dantedojo"
    assert parsed["actor"]["type"] == "model-training"
    assert parsed["timelineId"] == "main"


def test_evidence_emit_also_exports_decision_node(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "shared" / "nodes.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DANTEFORGE_DECISION_STORE", str(store))
    monkeypatch.setenv("DANTEFORGE_DECISION_SESSION_ID", "session-replay")
    monkeypatch.setenv("DANTEFORGE_DECISION_TIMELINE_ID", "timeline-1")
    EvidenceLogger.clear()

    logger = EvidenceLogger.get_instance(run_id="ignored")
    bundle = logger.emit({"type": "eval.complete", "data": {"score": 0.91}}, organ="eval")

    parsed = json.loads(store.read_text(encoding="utf-8").strip())
    assert parsed["sessionId"] == "session-replay"
    assert parsed["timelineId"] == "timeline-1"
    assert parsed["input"]["prompt"] == "eval.complete"
    assert parsed["evidenceRef"] == bundle.hash


def test_replay_env_nodes_chain(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "shared" / "nodes.jsonl"
    monkeypatch.setenv("DANTEFORGE_DECISION_STORE", str(store))
    monkeypatch.setenv("DANTEFORGE_DECISION_SESSION_ID", "session-replay")
    monkeypatch.setenv("DANTEFORGE_DECISION_TIMELINE_ID", "timeline-1")

    first = record_decision_node(
        project_root=tmp_path,
        session_id="ignored",
        actor_id="train",
        prompt="train.finetune.started",
        result={"ok": True},
        success=True,
    )
    second = record_decision_node(
        project_root=tmp_path,
        session_id="ignored",
        actor_id="promote",
        prompt="promote.passport.issued",
        result={"ok": True},
        success=True,
    )

    assert second["parentId"] == first["id"]
    assert second["prevHash"] == first["hash"]
