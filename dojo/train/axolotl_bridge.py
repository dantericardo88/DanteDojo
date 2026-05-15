"""Axolotl config bridge — generates axolotl YAML from DanteDojo FinetuneConfig.

Provides a higher-level bridge layer on top of axolotl_backend.generate_axolotl_config:
- Embeds dojo_passport_id for traceability
- Handles SpecializationPack metadata (dataset path, pack version, provenance)
- Writes YAML to disk and emits TRAIN_START evidence with bridge metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import HALProfile, HardwareTier
from dojo.pack.schema import SpecializationPack
from dojo.train.axolotl_backend import generate_axolotl_config
from dojo.train.base import FinetuneConfig


def axolotl_config_from_finetune(
    config: FinetuneConfig,
    pack: SpecializationPack,
    hal_profile: Optional[HALProfile] = None,
    passport_id: Optional[str] = None,
) -> dict[str, Any]:
    """Generate an axolotl YAML config dict from DanteDojo FinetuneConfig + SpecializationPack.

    Maps DanteDojo fields to axolotl fields:
        - base_model         → base_model
        - pack_id / data_path → datasets[0].path
        - lora_r / lora_alpha / lora_dropout → lora_*
        - num_epochs, batch_size, grad_accum_steps, learning_rate, warmup_ratio
        - max_seq_length     → sequence_len
        - output_dir         → output_dir
        - seed               → seed

    Embeds dojo_passport_id and pack provenance as a custom ``_dojo`` top-level key
    for traceability without polluting standard axolotl fields.

    Args:
        config:      DanteDojo FinetuneConfig
        pack:        SpecializationPack with data_path, pack_id, version, provenance
        hal_profile: Optional HALProfile; if None a CPU-tier profile is used (stub)
        passport_id: Optional ConstitutionalPassport.passport_id for traceability

    Returns:
        dict ready for yaml.dump() as an axolotl config
    """
    if hal_profile is None:
        # Create a minimal CPU-tier profile for config generation
        from datetime import datetime, timezone
        hal_profile = HALProfile(
            tier=HardwareTier.TIER_CPU,
            gpu_name="cpu",
            vram_bytes=0,
            compute_capability="0.0",
            has_native_fp8=False,
            has_native_fp4=False,
        )

    cfg = generate_axolotl_config(config, hal_profile)

    # Override dataset path with pack's actual data_path if available
    dataset_path = str(pack.data_path) if pack.data_path else pack.pack_id
    cfg["datasets"] = [
        {
            "path": dataset_path,
            "type": pack.format.value,
            "ds_type": "json",
        }
    ]

    # Embed DanteDojo provenance as a custom _dojo block
    cfg["_dojo"] = {
        "pack_id": pack.pack_id,
        "pack_version": pack.version,
        "schema_version": pack.schema_version,
        "harvest_receipt": pack.harvest_receipt,
        "dojo_passport_id": passport_id or "",
        "config_hash": config.config_hash(),
        "sample_count": pack.sample_count,
        "provenance": {
            "harvest_organ_version": pack.provenance.harvest_organ_version,
            "sources": list(pack.provenance.sources),
            "collection_date_utc": pack.provenance.collection_date_utc,
            "dedup_method": pack.provenance.dedup_method,
            "human_reviewed": pack.provenance.human_reviewed,
        },
    }

    return cfg


def export_axolotl_config(
    config: FinetuneConfig,
    pack: SpecializationPack,
    output_path: Path,
    hal_profile: Optional[HALProfile] = None,
    passport_id: Optional[str] = None,
) -> Path:
    """Write axolotl YAML config to output_path and emit TRAINING_START evidence.

    Args:
        config:      DanteDojo FinetuneConfig
        pack:        SpecializationPack
        output_path: Destination file path (parent dirs are created automatically)
        hal_profile: Optional HALProfile
        passport_id: Optional passport ID for traceability

    Returns:
        output_path (resolved)
    """
    try:
        import yaml  # type: ignore[import]
        _have_yaml = True
    except ImportError:
        _have_yaml = False

    cfg = axolotl_config_from_finetune(config, pack, hal_profile=hal_profile, passport_id=passport_id)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _have_yaml:
        with output_path.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    else:
        import json
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=False)

    # Emit TRAINING_START evidence with bridge metadata
    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.TRAINING_START.value,
            "data": {
                "bridge": "axolotl",
                "config_path": str(output_path),
                "pack_id": pack.pack_id,
                "pack_version": pack.version,
                "base_model": config.base_model,
                "config_hash": config.config_hash(),
                "dojo_passport_id": passport_id or "",
                "has_native_yaml": _have_yaml,
            },
        },
        organ="train",
    )

    return output_path


__all__ = [
    "axolotl_config_from_finetune",
    "export_axolotl_config",
]
