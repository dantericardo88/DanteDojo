"""DanteDojo CLI — single entry point for all dojo commands.

Usage:
    dojo detect
    dojo pack import <uri>
    dojo train <base-model>
    dojo serve <artifact-id>
    dojo eval <artifact-id>
    dojo promote <artifact-id>
    dojo registry list|lookup|register
    dojo experiment list
    dojo evidence chain|export
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="dojo",
    help="DanteDojo: sovereign AI training, evaluation, and inference organ.",
    no_args_is_help=True,
)

console = Console()


# ---------------------------------------------------------------------------
# HAL detection
# ---------------------------------------------------------------------------

@app.command()
def detect(
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Autodetect hardware and emit a signed HALProfile."""
    from dojo.hal.detector import detect as _detect
    profile = _detect()
    if json_out:
        typer.echo(json.dumps(profile.to_dict(), indent=2))
    else:
        d = profile.to_dict()
        console.print(f"[bold green]HAL Detection Result[/bold green]")
        console.print(f"  Tier:               {d['tier']}")
        console.print(f"  GPU:                {d['gpu_name']}")
        console.print(f"  VRAM:               {profile.vram_gb:.1f} GB")
        console.print(f"  Compute capability: {d['compute_capability']}")
        console.print(f"  Native FP8:         {d['has_native_fp8']}")
        console.print(f"  Driver:             {d['driver_version']}")
        console.print(f"  SHA-256:            {d['profile_sha256'][:16]}...")


# ---------------------------------------------------------------------------
# Pack management
# ---------------------------------------------------------------------------

pack_app = typer.Typer(help="Specialization pack commands.", no_args_is_help=True)
app.add_typer(pack_app, name="pack")


@pack_app.command("import")
def pack_import(
    uri: str = typer.Argument(..., help="Pack URI, e.g. harvest://empanada-bookkeeping/v3"),
    signature: str = typer.Option("", "--sig", help="Ed25519 signature hex (omit for test sentinel)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, do not index"),
) -> None:
    """Import a specialization pack from a Harvest URI."""
    from dojo.pack.importer import import_pack
    sig_bytes = bytes.fromhex(signature) if signature else b"test_signature"
    with console.status(f"Importing pack {uri!r}…"):
        result = import_pack(uri=uri, raw_data=[], signature=sig_bytes)
    d = result.to_dict() if hasattr(result, "to_dict") else vars(result)
    console.print(f"[green]Pack imported:[/green] {d.get('pack_id', uri)}")
    if json.dumps(d, default=str):
        console.print(json.dumps(d, indent=2, default=str))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.command()
def train(
    base_model: str = typer.Argument(..., help="HuggingFace model ID or local path"),
    pack_uri: str = typer.Option("", "--pack", help="Pack URI to train on"),
    epochs: int = typer.Option(1, "--epochs", help="Training epochs"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    output_dir: str = typer.Option("", "--output-dir", help="Checkpoint output directory"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Fine-tune a base model using the Dojo training pipeline."""
    from dojo.core.types import (
        DojoConfig, HardwareTier, PrecisionType, TrainingBackend, TrainingMode,
    )
    from dojo.hal.detector import detect as _detect
    from dojo.train.base import FinetuneConfig
    from dojo.train.finetune import finetune

    profile = _detect()
    cfg = FinetuneConfig(
        base_model=base_model,
        output_dir=Path(output_dir) if output_dir else Path("./checkpoints"),
        num_epochs=epochs,
        seed=seed,
    )
    with console.status(f"Training {base_model!r}…"):
        result = finetune(config=cfg, profile=profile)
    if json_out:
        typer.echo(json.dumps(result.to_dict() if hasattr(result, "to_dict") else vars(result), default=str, indent=2))
    else:
        console.print(f"[green]Training complete[/green]")
        console.print(f"  Receipt: {result.training_receipt[:16]}...")
        console.print(f"  Epochs:  {result.epochs_completed}")


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

@app.command()
def serve(
    artifact_id: str = typer.Argument(..., help="Dojo model artifact ID"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Launch an inference server for a registered model artifact."""
    from dojo.hal.detector import detect as _detect
    from dojo.serve.launcher import launch_server

    profile = _detect()
    with console.status(f"Launching server for {artifact_id!r}…"):
        manifest = launch_server(artifact_id=artifact_id, profile=profile, host=host, port=port)
    if json_out:
        typer.echo(json.dumps(manifest.to_dict() if hasattr(manifest, "to_dict") else vars(manifest), default=str, indent=2))
    else:
        d = manifest.to_dict() if hasattr(manifest, "to_dict") else vars(manifest)
        console.print(f"[green]Server ready[/green]  {d.get('endpoint_url', f'http://{host}:{port}')}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.command()
def eval(
    artifact_id: str = typer.Argument(..., help="Dojo model artifact ID to evaluate"),
    suite: str = typer.Option("dante", "--suite", help="Eval suite: dante|lm_eval|livebench"),
    tasks: str = typer.Option("", "--tasks", help="Comma-separated task names"),
    seeds: int = typer.Option(5, "--seeds", help="Number of seeds for CI"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run the evaluation harness on a registered model artifact."""
    from dojo.eval.harness import EvalSuite, run_eval

    task_list = tuple(t.strip() for t in tasks.split(",") if t.strip()) if tasks else ()
    suite_obj = EvalSuite(suite_type=suite, tasks=task_list, seeds=seeds)

    with console.status(f"Evaluating {artifact_id!r}…"):
        receipt = run_eval(artifact_id=artifact_id, suite=suite_obj)

    if json_out:
        typer.echo(json.dumps({
            "artifact_id": receipt.model_artifact_id,
            "mean_scores": receipt.mean_scores,
            "merkle_hash": receipt.merkle_hash,
            "contamination_ok": receipt.contamination_check_passed,
        }, indent=2))
    else:
        console.print(f"[green]Eval complete[/green]  merkle={receipt.merkle_hash[:16]}...")
        if receipt.mean_scores:
            t = Table("Task", "Mean Score", "95% CI Lower", "95% CI Upper")
            for task, score in receipt.mean_scores.items():
                t.add_row(task, f"{score:.4f}", f"{receipt.ci_lower.get(task, 0):.4f}", f"{receipt.ci_upper.get(task, 0):.4f}")
            console.print(t)


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

@app.command()
def promote(
    artifact_id: str = typer.Argument(..., help="Dojo model artifact ID to promote"),
    harvest_receipt_id: str = typer.Option("", "--harvest-receipt", help="Harvest receipt ID"),
    eval_merkle: str = typer.Option("", "--eval-merkle", help="Eval receipt Merkle hash"),
    baseline_f1: float = typer.Option(0.70, "--baseline-f1", help="Baseline F1 score"),
    candidate_f1: float = typer.Option(0.80, "--candidate-f1", help="Candidate F1 score"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run the three-way promotion gate to issue a ConstitutionalPassport."""
    from datetime import datetime, timezone

    from dojo.core.types import ModelArtifact
    from dojo.promote.gate import promote as _promote

    artifact = ModelArtifact(
        artifact_id=artifact_id,
        base_model="unknown",
        lora_weights=None,
        quantization=None,
        training_receipt=eval_merkle or "eval_receipt",
        passport=None,
    )
    with console.status(f"Running promotion gate for {artifact_id!r}…"):
        passport = _promote(
            candidate=artifact,
            harvest_receipt_id=harvest_receipt_id or f"harvest-{artifact_id}",
            eval_receipt_merkle=eval_merkle or "eval_receipt",
            baseline_f1=baseline_f1,
            candidate_f1=candidate_f1,
        )

    if json_out:
        typer.echo(json.dumps(passport.to_dict(), default=str, indent=2))
    else:
        console.print(f"[bold green]Passport issued[/bold green]")
        console.print(f"  Passport ID: {passport.passport_id}")
        console.print(f"  Expires:     {passport.expires_at.isoformat()}")
        console.print(f"  Merkle root: {passport.merkle_root[:16]}...")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

registry_app = typer.Typer(help="Model registry commands.", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


@registry_app.command("list")
def registry_list() -> None:
    """List all artifacts registered in the local Dojo registry."""
    from dojo.registry.store import LocalRegistryStore
    store = LocalRegistryStore()
    artifacts = store.list_artifacts()
    if not artifacts:
        console.print("[yellow]No artifacts registered.[/yellow]")
        return
    t = Table("Artifact ID", "Base Model", "Created At")
    for a in artifacts:
        t.add_row(a.artifact_id, a.base_model, a.created_at.isoformat())
    console.print(t)


@registry_app.command("lookup")
def registry_lookup(
    artifact_id: str = typer.Argument(..., help="Artifact ID to look up"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Look up a model artifact by ID."""
    from dojo.registry.registry import lookup
    artifact = lookup(artifact_id=artifact_id)
    if json_out:
        typer.echo(json.dumps(artifact.to_dict(), default=str, indent=2))
    else:
        d = artifact.to_dict()
        for k, v in d.items():
            console.print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

experiment_app = typer.Typer(help="Experiment tracking commands.", no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("list")
def experiment_list() -> None:
    """List all runs in the local experiment tracker."""
    from dojo.experiment.tracker import ExperimentTracker
    tracker = ExperimentTracker()
    runs = tracker.list_runs()
    if not runs:
        console.print("[yellow]No experiment runs recorded.[/yellow]")
        return
    t = Table("Run ID", "Name", "Status", "Started")
    for r in runs:
        t.add_row(r.get("run_id", ""), r.get("name", ""), r.get("status", ""), r.get("start_time", ""))
    console.print(t)


# ---------------------------------------------------------------------------
# Evidence chain
# ---------------------------------------------------------------------------

evidence_app = typer.Typer(help="Evidence chain commands.", no_args_is_help=True)
app.add_typer(evidence_app, name="evidence")


@evidence_app.command("chain")
def evidence_chain(
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Display the current Merkle-chained evidence summary."""
    from dojo.core.evidence import EvidenceLogger
    logger = EvidenceLogger.get_instance()
    chain = logger.chain()
    if json_out:
        typer.echo(json.dumps({"run_id": chain.run_id, "size": chain.size, "merkle_root": chain.merkle_root}, indent=2))
    else:
        console.print(f"[bold]Evidence Chain[/bold]")
        console.print(f"  Run ID:      {chain.run_id}")
        console.print(f"  Events:      {chain.size}")
        console.print(f"  Merkle root: {chain.merkle_root[:32]}..." if chain.merkle_root else "  Merkle root: (empty)")


@evidence_app.command("export")
def evidence_export(
    output: Path = typer.Argument(..., help="Output path for the JSON receipt file"),
) -> None:
    """Export the full evidence receipt to a JSON file."""
    from dojo.core.evidence import EvidenceLogger
    logger = EvidenceLogger.get_instance()
    receipt = logger.export_receipt(output)
    console.print(f"[green]Receipt exported[/green] → {output}")
    console.print(f"  Merkle root: {receipt['merkle_root'][:32]}...")
    console.print(f"  Events:      {receipt['event_count']}")


# ---------------------------------------------------------------------------
# Layer streaming info
# ---------------------------------------------------------------------------

stream_app = typer.Typer(help="Layer streaming utilities.", no_args_is_help=True)
app.add_typer(stream_app, name="stream")


@stream_app.command("status")
def stream_status() -> None:
    """Show layer streaming module status (mock_mode / torch availability)."""
    try:
        import torch  # type: ignore[import-not-found]
        torch_ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        torch_ver = "not installed"
        cuda_ok = False
    console.print(f"[bold]Stream Layer Status[/bold]")
    console.print(f"  torch:       {torch_ver}")
    console.print(f"  CUDA:        {cuda_ok}")
    console.print(f"  Mock mode:   {not cuda_ok}")


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the DanteDojo version."""
    try:
        from importlib.metadata import version as _v
        v = _v("dante-dojo")
    except Exception:
        v = "0.1.0-dev"
    console.print(f"[bold]DanteDojo[/bold] v{v}")


if __name__ == "__main__":
    app()
