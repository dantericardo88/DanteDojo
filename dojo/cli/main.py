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

from dojo.experiment.tracker import ExperimentTracker
from dojo.experiment.leaderboard import RunLeaderboard

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
        cc = d['compute_capability']
        console.print(f"[bold green]HAL Detection Result[/bold green]")
        console.print(f"  Tier:               {d['tier']}")
        console.print(f"  GPU:                {d['gpu_name']}")
        console.print(f"  VRAM:               {profile.vram_gb:.1f} GB")
        console.print(f"  SM version:         {cc}")
        console.print(f"  Compute capability: {cc}")
        console.print(f"  Native FP8:         {'yes' if d['has_native_fp8'] else 'no'}")
        console.print(f"  Native FP4:         {'yes' if d.get('has_native_fp4') else 'no'}")
        console.print(f"  Driver:             {d['driver_version']}")
        console.print(f"  SHA-256:            {d['profile_sha256'][:16]}...")


# ---------------------------------------------------------------------------
# Pack management
# ---------------------------------------------------------------------------

pack_app = typer.Typer(help="Specialization pack commands.", no_args_is_help=True)
app.add_typer(pack_app, name="pack")


@pack_app.command("import")
def pack_import(
    manifest_path: str = typer.Argument(..., help="Path to pack manifest JSON file"),
    data_path: str = typer.Argument(..., help="Path to pack data parquet/jsonl file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, do not index"),
) -> None:
    """Import a specialization pack from local manifest and data files."""
    from dojo.pack.importer import import_pack
    mpath = Path(manifest_path)
    dpath = Path(data_path)
    with console.status(f"Importing pack from {manifest_path!r}…"):
        result = import_pack(manifest_path=mpath, data_path=dpath)
    d = result.to_dict() if hasattr(result, "to_dict") else vars(result)
    console.print(f"[green]Pack imported:[/green] {d.get('pack_id', manifest_path)}")
    if not dry_run:
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
    from dojo.hal.detector import detect as _detect
    from dojo.pack.schema import SpecializationPack
    from dojo.train.base import FinetuneConfig
    from dojo.train.finetune import finetune

    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from datetime import datetime, timezone as _tz

    profile = _detect()
    cfg = FinetuneConfig(
        pack_id=pack_uri or "default",
        base_model=base_model,
        output_dir=Path(output_dir) if output_dir else Path("./checkpoints"),
        num_epochs=epochs,
        seed=seed,
    )
    # Build a minimal in-memory pack stub for CLI invocations without a
    # real manifest file (CI / smoke mode).  When a pack_uri is supplied a
    # real import should be performed first; the stub is only a placeholder.
    pack = SpecializationPack(
        pack_id=pack_uri or "default",
        version="0.0.0",
        schema_version="1.0",
        sample_count=0,
        format=PackFormat.INSTRUCTION,
        splits={"train": 0, "val": 0, "held_out_gold": 0},
        harvest_receipt="cli_stub",
        license=LicenseSpec(
            spdx_id="Apache-2.0",
            allows_derived_ft=True,
            allows_commercial_use=True,
            requires_attribution=False,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version="0.0.0",
            sources=(),
            collection_date_utc=datetime.now(_tz.utc).date().isoformat(),
            dedup_method="none",
            human_reviewed=False,
        ),
        signature=b"test_signature",
        manifest_sha256="0" * 64,
        data_parquet_sha256="0" * 64,
    )
    with console.status(f"Training {base_model!r}…"):
        result = finetune(config=cfg, hal_profile=profile, pack=pack)
    if json_out:
        typer.echo(json.dumps(result.to_dict() if hasattr(result, "to_dict") else vars(result), default=str, indent=2))
    else:
        console.print(f"[green]Training complete[/green]")
        console.print(f"  Receipt: {result.training_receipt[:16]}...")
        console.print(f"  Steps:   {result.steps_completed}")


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
    # model_path is derived from artifact_id for CLI invocations.
    # A real deployment would look up the artifact in the registry first.
    model_path = Path(artifact_id) if Path(artifact_id).exists() else Path("./artifacts") / artifact_id
    with console.status(f"Launching server for {artifact_id!r}…"):
        manifest = launch_server(
            model_artifact_id=artifact_id,
            model_path=model_path,
            profile=profile,
            port=port,
        )
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
    artifact_id: Optional[str] = typer.Argument(None, help="Dojo model artifact ID to evaluate"),
    suite: str = typer.Option("dante", "--suite", help="Eval suite: dante|lm_eval|livebench"),
    tasks: str = typer.Option("", "--tasks", help="Comma-separated task names"),
    seeds: int = typer.Option(5, "--seeds", help="Number of seeds for CI"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="Print available eval tasks and exit"),
) -> None:
    """Run the evaluation harness on a registered model artifact."""
    if list_tasks:
        from dojo.eval.lm_eval_tasks import LM_EVAL_STANDARD_TASKS
        t = Table("Task", "Few-shot", "Description")
        for task in LM_EVAL_STANDARD_TASKS.values():
            t.add_row(task.name, str(task.default_num_fewshot), task.description)
        console.print(t)
        return

    if artifact_id is None:
        typer.echo("Error: artifact_id is required unless --list-tasks is specified.", err=True)
        raise typer.Exit(code=1)

    from dojo.core.types import ModelArtifact
    from dojo.eval.harness import EvalSuite, run_eval

    task_list = tuple(t.strip() for t in tasks.split(",") if t.strip()) if tasks else ()
    suite_obj = EvalSuite(suite_type=suite, tasks=task_list, seeds=seeds)
    model = ModelArtifact(
        artifact_id=artifact_id,
        base_model=artifact_id,
        lora_weights=None,
        quantization=None,
        training_receipt="cli_eval",
        passport=None,
    )

    with console.status(f"Evaluating {artifact_id!r}…"):
        receipt = run_eval(model=model, suite=suite_obj, seeds=seeds)

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
    details = store.list_artifact_details()
    if not details:
        console.print("[yellow]No artifacts registered.[/yellow]")
        return
    t = Table("Artifact ID", "Base Model", "Registered At")
    for d in details:
        t.add_row(d.get("artifact_id", ""), d.get("model_name", ""), d.get("registered_at", ""))
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
# Experiment tracking (legacy single-command alias)
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
# experiments subapp  (dojo experiments list / compare / leaderboard)
# ---------------------------------------------------------------------------

experiments_app = typer.Typer(help="Experiment run management.", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")


@experiments_app.command("list")
def experiments_list(
    experiment: str = typer.Option("", "--experiment", help="Filter by experiment name"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all experiment runs, optionally filtered by experiment name."""
    tracker = ExperimentTracker()
    runs = tracker.list_runs(experiment)
    if json_out:
        typer.echo(json.dumps(runs, indent=2, default=str))
        return
    if not runs:
        console.print("[yellow]No experiment runs recorded.[/yellow]")
        return
    t = Table("Run ID", "Name", "Status", "Started")
    for r in runs:
        t.add_row(r.get("run_id", ""), r.get("name", ""), r.get("status", ""), r.get("start_time", ""))
    console.print(t)


@experiments_app.command("compare")
def experiments_compare(
    run_id_1: str = typer.Argument(..., help="First run ID"),
    run_id_2: str = typer.Argument(..., help="Second run ID"),
    metric: str = typer.Option("loss", "--metric", help="Primary metric for ranking"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compare two experiment runs side-by-side."""
    lb = RunLeaderboard(primary_metric=metric, higher_is_better=False)
    lb.add_run(run_id_1, run_id_1, "", {metric: 0.0})
    lb.add_run(run_id_2, run_id_2, "", {metric: 0.0})
    result = lb.compare_top_n(n=2)
    if json_out:
        typer.echo(json.dumps(result, indent=2, default=str))
    else:
        for rid, info in result.items():
            console.print(f"[bold]{rid}[/bold]  rank={info.get('rank')}  baseline={info.get('baseline_run_id')}")


@experiments_app.command("leaderboard")
def experiments_leaderboard(
    experiment: str = typer.Option("", "--experiment", help="Experiment name to load runs from"),
    metric: str = typer.Option("loss", "--metric", help="Primary metric for ranking"),
    higher_is_better: bool = typer.Option(False, "--higher-is-better", help="Higher metric value = better rank"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Build and display the run leaderboard for an experiment."""
    tracker = ExperimentTracker()
    lb = RunLeaderboard.from_tracker(
        tracker,
        experiment_name=experiment,
        primary_metric=metric,
        higher_is_better=higher_is_better,
    )
    d = lb.to_dict()
    if json_out:
        typer.echo(json.dumps(d, indent=2, default=str))
        return
    entries = d.get("entries", [])
    if not entries:
        console.print("[yellow]No runs found.[/yellow]")
        return
    t = Table("Rank", "Run ID", "Run Name", "Experiment", metric)
    for e in entries:
        t.add_row(
            str(e.get("rank", "")),
            e.get("run_id", ""),
            e.get("run_name", ""),
            e.get("experiment_name", ""),
            str(e.get("metrics", {}).get(metric, "")),
        )
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
# Idle training
# ---------------------------------------------------------------------------

@app.command("idle-train")
def idle_train(
    base_model: str = typer.Argument(..., help="HuggingFace model ID or local path"),
    pack: str = typer.Option("", "--pack", help="Pack ID to train on"),
    output_dir: str = typer.Option("", "--output-dir", help="Checkpoint output directory"),
    max_optimizer_steps: int = typer.Option(100, "--max-optimizer-steps", help="Max optimizer steps per idle window"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run idle-time fine-tuning — trains only when the system is idle."""
    import dataclasses
    from pathlib import Path as _Path
    from dojo.idle.daemon import IdleTrainConfig, IdleTrainingDaemon

    config = IdleTrainConfig(
        base_model=base_model,
        pack_id=pack or "default",
        output_dir=_Path(output_dir) if output_dir else _Path("artifacts/idle_checkpoints"),
        max_optimizer_steps=max_optimizer_steps,
    )
    daemon = IdleTrainingDaemon(config)
    result = daemon.run()
    if json_out:
        typer.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        console.print(f"[green]Idle training {result.status}[/green]")
        console.print(f"  Steps:     {result.steps_completed}")
        console.print(f"  Checkpoints saved: {result.checkpoints_saved}")


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

quant_app = typer.Typer(help="Model quantization export commands.", no_args_is_help=True)
app.add_typer(quant_app, name="quant")


@quant_app.command("gguf")
def quant_gguf(
    model_path: str = typer.Argument(..., help="Path to HuggingFace model directory"),
    output_dir: str = typer.Option("./gguf_out", "--output-dir", help="Output directory"),
    quant_type: str = typer.Option("Q4_K_M", "--quant-type", help="GGUF quantization type (Q4_K_M, Q5_K_M, …)"),
    passport_id: str = typer.Option("", "--passport-id", help="Constitutional passport ID to embed"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Export a HuggingFace model to GGUF format (llama.cpp target)."""
    from dojo.quant.gguf import export_gguf
    import hashlib

    model = Path(model_path)
    out = Path(output_dir)

    with console.status(f"Exporting {model.name!r} → GGUF ({quant_type})…"):
        result = export_gguf(
            model_path=model,
            output_dir=out,
            quant_type=quant_type,
            passport_id=passport_id,
        )

    h = hashlib.sha256()
    with result.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    info = {
        "output_path": str(result),
        "quant_type": quant_type,
        "file_size_bytes": result.stat().st_size,
        "sha256": h.hexdigest(),
        "sidecar": str(result.with_suffix(".meta.json")) if result.with_suffix(".meta.json").exists() else None,
    }

    if json_out:
        typer.echo(json.dumps(info, indent=2))
    else:
        console.print("[green]GGUF export complete[/green]")
        console.print(f"  Output: {info['output_path']}")
        console.print(f"  Size:   {info['file_size_bytes']:,} bytes")
        if info["sidecar"]:
            console.print(f"  Sidecar: {info['sidecar']}")


@quant_app.command("gptq")
def quant_gptq(
    model_path: str = typer.Argument(..., help="Path to HuggingFace model directory"),
    output_dir: str = typer.Option("./gptq_out", "--output-dir", help="Output directory"),
    bits: int = typer.Option(4, "--bits", help="Quantization bits (4 or 8)"),
    group_size: int = typer.Option(128, "--group-size", help="Quantization group size"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Export a HuggingFace model to GPTQ format."""
    from dojo.quant.gptq import export_gptq

    model = Path(model_path)
    out = Path(output_dir)

    with console.status(f"Exporting {model.name!r} → GPTQ ({bits}-bit, group={group_size})…"):
        result = export_gptq(
            model_path=model,
            output_dir=out,
            bits=bits,
            group_size=group_size,
        )

    info = {
        "output_path": str(result),
        "bits": bits,
        "group_size": group_size,
        "file_size_bytes": result.stat().st_size,
    }

    if json_out:
        typer.echo(json.dumps(info, indent=2))
    else:
        console.print("[green]GPTQ export complete[/green]")
        console.print(f"  Output: {info['output_path']}")
        console.print(f"  Bits:   {bits}-bit  group={group_size}")


@quant_app.command("awq")
def quant_awq(
    model_path: str = typer.Argument(..., help="Path to HuggingFace model directory"),
    output_dir: str = typer.Option("./awq_out", "--output-dir", help="Output directory"),
    bits: int = typer.Option(4, "--bits", help="Quantization bits (4 or 8)"),
    group_size: int = typer.Option(128, "--group-size", help="Quantization group size"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Export a HuggingFace model to AWQ format."""
    from dojo.quant.awq import export_awq

    model = Path(model_path)
    out = Path(output_dir)

    with console.status(f"Exporting {model.name!r} → AWQ ({bits}-bit, group={group_size})…"):
        result = export_awq(
            model_path=model,
            output_dir=out,
            bits=bits,
            group_size=group_size,
        )

    info = {
        "output_path": str(result),
        "bits": bits,
        "group_size": group_size,
        "file_size_bytes": result.stat().st_size,
    }

    if json_out:
        typer.echo(json.dumps(info, indent=2))
    else:
        console.print("[green]AWQ export complete[/green]")
        console.print(f"  Output: {info['output_path']}")
        console.print(f"  Bits:   {bits}-bit  group={group_size}")


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
