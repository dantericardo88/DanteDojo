# DanteDojo Hardware Tiers

## Tier Definitions

| Enum                   | Value               | GPU                          | VRAM  | SM   | Notes                        |
|------------------------|---------------------|------------------------------|-------|------|------------------------------|
| `TIER_4060_LAPTOP`     | `tier_4060_laptop`  | RTX 4060 Laptop GPU          | 8 GB  | 8.9  | Ada, native FP8, NF4 QLoRA   |
| `TIER_3090`            | `tier_3090`         | RTX 3090                     | 24 GB | 8.6  | Ampere, **no native FP8**    |
| `TIER_3090_DUAL`       | `tier_3090_dual`    | 2× RTX 3090                  | 48 GB | 8.6  | vLLM TP=2                    |
| `TIER_H100`            | `tier_h100`         | H100 SXM / PCIe              | 80 GB | 9.0  | Hopper, native FP8           |
| `TIER_APPLE_M4`        | `tier_apple_m4`     | Apple M4 (unified)           | var   | N/A  | MLX backend                  |
| `TIER_AMD_MI300`       | `tier_amd_mi300`    | AMD MI300X                   | 192 GB| N/A  | ROCm, native FP8             |
| `TIER_INTEL_B580`      | `tier_intel_b580`   | Intel Arc B580               | 12 GB | N/A  | ARC, limited precision       |
| `TIER_CPU`             | `tier_cpu`          | None                         | —     | N/A  | Inference only               |

## Precision Matrix

| Tier          | BF16 | FP8 (native) | NF4 | W8A16-Marlin | INT4 |
|---------------|------|-------------|-----|-------------|------|
| 4060 Laptop   | ✓    | ✓           | ✓   | ✓           | ✓    |
| 3090          | ✓    | **✗** (SM 8.6 < 8.9) | ✓ | ✓ | ✓ |
| 3090 Dual     | ✓    | **✗**       | ✓   | ✓           | ✓    |
| H100          | ✓    | ✓           | ✓   | ✓           | ✓    |
| Apple M4      | ✓    | ✗           | ✓   | ✗           | ✓    |
| CPU           | ✓    | ✗           | ✗   | ✗           | ✓    |

### Ampere FP8 Rejection (Critical)

The RTX 3090 is SM 8.6 (Ampere). FP8 compute requires SM ≥ 8.9 (Ada/Hopper/Blackwell). Ampere can store weights in FP8 format but executes them as BF16 — causing silent accuracy loss with no throughput benefit.

`dojo/precision/fp8.py` hard-rejects FP8 on any profile where `float(compute_capability) < 8.9`:

```python
# raises PrecisionUnsupported immediately — never silently degrade
FP8Handler.ensure_supported(profile)
```

`dojo/turbo/policy.py` maps Ampere to `[BF16_DEFAULT, W8A16_MARLIN_SERVING]` — FP8 is never in the policy list.

## Training Backend Selection

Detection order in `dojo/train/finetune.py`:

```
TIER_4060_LAPTOP → Unsloth
TIER_3090        → Unsloth
TIER_3090_DUAL   → Axolotl  (FSDP2, TP=2)
TIER_H100        → Axolotl / NATIVE
TIER_APPLE_M4    → MLX
TIER_CPU         → BackendUnavailable (training blocked)
```

## Inference Backend Selection

Detection order in `dojo/serve/launcher.py`:

```
TIER_4060_LAPTOP → llama.cpp (edge, quantized)
TIER_3090        → vLLM
TIER_3090_DUAL   → vLLM (TP=2)
TIER_H100        → vLLM
TIER_APPLE_M4    → MLX-LM
TIER_CPU         → llama.cpp
```

## VRAM Budget (4060 Laptop, 8 GB)

With 8 GB VRAM and an 85% safety ceiling (6.8 GB usable):

| Model            | Mode    | VRAM estimate | Fits? |
|------------------|---------|---------------|-------|
| Qwen2.5-7B       | NF4     | ~5.5 GB       | ✓     |
| Qwen2.5-7B       | BF16    | ~14 GB        | ✗     |
| Llama-3.2-3B     | BF16    | ~6.5 GB       | ✓     |
| Llama-3.1-8B     | NF4     | ~5.5 GB       | ✓     |

`finetune()` rejects configs that exceed 85% of `profile.vram_bytes` before any training begins.

## FinetuneConfig 4060-Safe Defaults

```python
FinetuneConfig(
    batch_size=1,
    grad_accum_steps=8,    # effective batch = 8
    max_seq_length=2048,
    gradient_checkpointing=True,
)
```
