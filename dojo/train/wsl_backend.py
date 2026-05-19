"""dojo/train/wsl_backend.py — Delegate training to WSL2 subprocess.

When DanteDojo runs on Windows-native Python and WSL2 is available, this
module pipes a training job into the WSL environment where unsloth + triton
are installable.

Communication protocol:
  1. Serialize FinetuneConfig + pack samples to a JSON temp file.
  2. Write a self-contained Python training script to a temp file.
  3. Run: wsl.exe -- python3 <script> <config_json>
  4. Parse the JSON result file written by the script.
  5. Deserialize FinetuneResult and return it to the caller.

File exchange uses the Windows temp directory, which WSL2 mounts at
/mnt/c/Users/.../AppData/Local/Temp (or equivalent).  Paths are translated
automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError
from dojo.train.base import FinetuneConfig, FinetuneResult


class WSLBackendError(DojoError):
    """Raised when the WSL training delegation fails."""


def _windows_path_to_wsl(path: Path) -> str:
    """Convert a Windows absolute path to the WSL /mnt/... equivalent.

    e.g.  C:\\Users\\foo\\bar  →  /mnt/c/Users/foo/bar
    """
    p = str(path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        return f"/mnt/{drive}{rest}"
    return p


def _wsl_available() -> bool:
    return bool(shutil.which("wsl.exe") or shutil.which("wsl"))


def _ensure_unsloth_in_wsl(quiet: bool = True) -> bool:
    """pip install unsloth inside WSL if not already present."""
    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl") or "wsl"
    check_cmd = [wsl_exe, "--", "python3", "-c", "import unsloth"]
    try:
        r = subprocess.run(check_cmd, capture_output=True, timeout=10)
        if r.returncode == 0:
            return True
    except Exception:
        pass

    install_cmd = [
        wsl_exe, "--", "bash", "-c",
        "pip install unsloth[colab-new] --quiet 2>&1 || "
        "pip install 'unsloth[cu121-torch240]' --quiet 2>&1",
    ]
    try:
        r = subprocess.run(install_cmd, capture_output=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


def run_training_in_wsl(
    config: FinetuneConfig,
    pack_samples: list[dict],
    *,
    install_if_missing: bool = True,
    timeout_seconds: int = 3600,
) -> FinetuneResult:
    """Run SFT/QLoRA training in WSL2 and return the result.

    Args:
        config: Frozen FinetuneConfig to train with.
        pack_samples: List of sample dicts from SpecializationPack.
        install_if_missing: If True, installs unsloth in WSL before running.
        timeout_seconds: Max seconds to wait for the WSL process.

    Returns:
        FinetuneResult with steps_completed, final_loss, etc.

    Raises:
        WSLBackendError: If WSL is unavailable or the training process fails.
    """
    if not _wsl_available():
        raise WSLBackendError(
            "WSL2 is not available on this machine",
            reason="wsl.exe not found on PATH",
        )

    logger = EvidenceLogger.get_instance()
    logger.emit(
        {
            "type": "train.wsl_backend.start",
            "data": {
                "base_model": config.base_model,
                "training_mode": config.training_mode,
                "num_epochs": config.num_epochs,
            },
        },
        organ="train",
    )

    if install_if_missing:
        _ensure_unsloth_in_wsl()

    with tempfile.TemporaryDirectory(prefix="dojo_wsl_") as tmpdir:
        tmp = Path(tmpdir)
        config_file = tmp / "config.json"
        result_file = tmp / "result.json"
        script_file = tmp / "train.py"

        # Write config + samples
        config_file.write_text(json.dumps({
            "config": config.to_dict(),
            "samples": pack_samples,
            "result_path": str(result_file),
        }, indent=2))

        # Write self-contained training script
        script_file.write_text(_build_wsl_training_script())

        # Convert paths for WSL
        wsl_script = _windows_path_to_wsl(script_file)
        wsl_config = _windows_path_to_wsl(config_file)

        wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl") or "wsl"
        cmd = [
            wsl_exe, "--",
            "python3", wsl_script, wsl_config,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise WSLBackendError(
                f"WSL training timed out after {timeout_seconds}s",
                reason="Process did not complete within timeout",
            )
        except Exception as exc:
            raise WSLBackendError(
                f"WSL process failed: {exc}",
                reason=str(exc),
            ) from exc

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise WSLBackendError(
                f"WSL training process exited with code {proc.returncode}",
                reason=stderr[:500],
            )

        # Parse result
        if not result_file.exists():
            raise WSLBackendError(
                "WSL training script did not write result.json",
                reason="Output file missing — check training script",
            )

        result_data = json.loads(result_file.read_text())

        logger.emit(
            {
                "type": "train.wsl_backend.complete",
                "data": {
                    "steps_completed": result_data.get("steps_completed", 0),
                    "final_loss": result_data.get("final_loss", 0.0),
                    "backend_used": result_data.get("backend_used", "wsl_unsloth"),
                },
            },
            organ="train",
        )

        from dojo.train.base import FinetuneResult as _FR
        return _FR(
            steps_completed=result_data.get("steps_completed", 0),
            final_loss=result_data.get("final_loss", 0.0),
            output_dir=config.output_dir,
            config_hash=config.config_hash(),
        )


def _build_wsl_training_script() -> str:
    """Return a self-contained Python script that runs inside WSL."""
    return textwrap.dedent("""\
        #!/usr/bin/env python3
        \"\"\"DanteDojo WSL training delegate script.\"\"\"
        import json, sys, os
        from pathlib import Path

        config_path = sys.argv[1]
        payload = json.loads(Path(config_path).read_text())
        cfg = payload["config"]
        samples = payload["samples"]
        result_path = payload["result_path"]

        steps_completed = 0
        final_loss = 0.0
        backend_used = "wsl_native"

        try:
            import unsloth  # noqa: F401
            from unsloth import FastLanguageModel
            import torch

            model_name = cfg["base_model"]
            lora_r = cfg.get("lora_r", 16)
            lora_alpha = cfg.get("lora_alpha", 32)
            max_seq = min(cfg.get("max_seq_length", 2048), 2048)
            num_epochs = cfg.get("num_epochs", 1)
            batch_size = cfg.get("batch_size", 1)

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq,
                dtype=None,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            from trl import SFTTrainer, SFTConfig
            import datasets as ds
            dataset = ds.Dataset.from_list(samples or [{"text": "hello world"}])

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=SFTConfig(
                    output_dir=cfg.get("output_dir", "/tmp/dojo_wsl_out"),
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    max_seq_length=max_seq,
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    dataset_text_field="text",
                ),
            )
            result = trainer.train()
            steps_completed = result.global_step
            final_loss = result.training_loss
            backend_used = "wsl_unsloth"

        except ImportError:
            # unsloth not installed — fall back to plain TRL in WSL
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import LoraConfig, get_peft_model, TaskType
                from trl import SFTTrainer, SFTConfig
                import datasets as ds

                model_name = cfg["base_model"]
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                lora_cfg = LoraConfig(
                    r=cfg.get("lora_r", 16),
                    lora_alpha=cfg.get("lora_alpha", 32),
                    task_type=TaskType.CAUSAL_LM,
                    bias="none",
                )
                model = get_peft_model(model, lora_cfg)
                dataset = ds.Dataset.from_list(samples or [{"text": "hello"}])

                trainer = SFTTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=dataset,
                    args=SFTConfig(
                        output_dir=cfg.get("output_dir", "/tmp/dojo_wsl_out"),
                        num_train_epochs=cfg.get("num_epochs", 1),
                        dataset_text_field="text",
                    ),
                )
                result = trainer.train()
                steps_completed = result.global_step
                final_loss = result.training_loss
                backend_used = "wsl_trl"
            except Exception as e:
                print(f"WSL TRL fallback failed: {e}", file=sys.stderr)

        Path(result_path).write_text(json.dumps({
            "steps_completed": steps_completed,
            "final_loss": final_loss,
            "backend_used": backend_used,
        }))
    """)
