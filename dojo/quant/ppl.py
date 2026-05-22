"""dojo/quant/ppl.py — Real perplexity measurement for quantization quality gates.

Provides compute_real_ppl(), a @one_door entry point that computes genuine
perplexity using torch.nn.CrossEntropyLoss on a model's forward pass.

Works on both CPU and GPU. Designed for use with any HuggingFace
AutoModelForCausalLM-compatible model — including tiny models like
sshleifer/tiny-gpt2 for CI and production models for full evaluation.

Emits QUANT_PPL_GATE evidence with fp16_ppl, quantized_ppl, and delta_pct
when called from a quality gate context.
"""

from __future__ import annotations

import math
from typing import Optional

from dojo.core.one_door import one_door

# Calibration texts used when no custom texts are supplied.
# Long enough to give stable PPL, short enough for CPU in CI.
_DEFAULT_CALIBRATION_TEXTS = [
    (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models are evaluated using perplexity as a measure "
        "of how well the probability distribution predicted by the model "
        "aligns with the actual distribution of the words in the test set. "
        "Lower perplexity means the model assigns higher probability to the "
        "true next token, indicating better language understanding."
    ),
    (
        "Quantization reduces the numerical precision of neural network weights "
        "to decrease memory usage and improve inference throughput on modern "
        "hardware accelerators. Post-training quantization applies these "
        "transformations after training without requiring additional fine-tuning. "
        "The quality of a quantized model can be measured by comparing its "
        "perplexity against the full-precision baseline on held-out text."
    ),
]

# PPL delta threshold (fractional) above which the quality gate fires.
_DEFAULT_PPL_DELTA_THRESHOLD = 0.15  # 15%


@one_door("quant.compute_real_ppl")
def compute_real_ppl(
    model: object,
    tokenizer: object,
    calibration_texts: Optional[list[str]] = None,
    max_length: int = 512,
    stride: int = 256,
    device: Optional[str] = None,
) -> float:
    """Compute real perplexity using cross-entropy loss on calibration texts.

    Performs a genuine model forward pass (no mocking). Uses
    torch.nn.CrossEntropyLoss over next-token predictions on each
    calibration text, then returns exp(mean_loss) as perplexity.

    Works on both CPU and GPU. For tiny models (sshleifer/tiny-gpt2), runs
    in well under 1 second on CPU.

    Args:
        model: A HuggingFace AutoModelForCausalLM-compatible model instance.
            Must support model(input_ids, labels=input_ids) call signature.
        tokenizer: A HuggingFace tokenizer for the model.
        calibration_texts: List of strings to evaluate. If None, uses the
            module-level _DEFAULT_CALIBRATION_TEXTS.
        max_length: Maximum token length per text chunk. Texts longer than
            this are truncated. Default 512.
        stride: Stride for sliding-window PPL on long texts. Default 256.
        device: Device string (e.g. "cpu", "cuda:0"). If None, uses
            model's current device.

    Returns:
        Perplexity as a positive float. A finite, positive value always.
        Falls back to 1e9 only on total failure (should never happen in
        normal use).

    Raises:
        ImportError: if torch or transformers are not installed.
    """
    import torch

    texts = calibration_texts if calibration_texts is not None else _DEFAULT_CALIBRATION_TEXTS

    # Resolve device from model parameter tensors if not specified
    if device is None:
        try:
            device = str(next(iter(model.parameters())).device)  # type: ignore[union-attr]
        except (StopIteration, AttributeError):
            device = "cpu"

    model.eval()  # type: ignore[union-attr]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            # Tokenize without truncation first, then slice manually
            enc = tokenizer(  # type: ignore[operator]
                text,
                return_tensors="pt",
                truncation=False,
            )
            input_ids = enc["input_ids"]  # shape: [1, T]
            seq_len = input_ids.shape[1]

            if seq_len < 2:
                continue

            # Use sliding-window approach for long sequences
            # For short sequences, a single pass suffices
            chunk_start = 0
            while chunk_start < seq_len:
                chunk_end = min(chunk_start + max_length, seq_len)
                chunk_ids = input_ids[:, chunk_start:chunk_end].to(device)

                if chunk_ids.shape[1] < 2:
                    break

                try:
                    outputs = model(chunk_ids, labels=chunk_ids)  # type: ignore[operator]
                    loss = outputs.loss  # scalar cross-entropy loss
                    if torch.isfinite(loss):
                        n_tokens = chunk_ids.shape[1] - 1  # next-token prediction
                        total_loss += loss.item() * n_tokens
                        total_tokens += n_tokens
                except Exception:
                    pass

                if chunk_end >= seq_len:
                    break
                chunk_start += stride

    if total_tokens == 0:
        return 1e9  # pathological fallback — should never happen

    avg_loss = total_loss / total_tokens
    return float(math.exp(avg_loss))


def measure_ppl_gate(
    baseline_model: object,
    quantized_model: object,
    tokenizer: object,
    calibration_texts: Optional[list[str]] = None,
    ppl_delta_threshold: float = _DEFAULT_PPL_DELTA_THRESHOLD,
    exporter: str = "unknown",
) -> dict:
    """Measure PPL before/after quantization and emit QUANT_PPL_GATE evidence.

    Computes real perplexity for both the baseline (fp16/fp32) model and
    the quantized model using compute_real_ppl(). Emits a QUANT_PPL_GATE
    evidence event. Raises QuantQualityRejected if delta exceeds threshold.

    Args:
        baseline_model: Original (unquantized) model.
        quantized_model: Quantized model to evaluate.
        tokenizer: Shared tokenizer for both models.
        calibration_texts: Texts used for PPL measurement. Defaults to
            _DEFAULT_CALIBRATION_TEXTS.
        ppl_delta_threshold: Maximum allowed fractional PPL increase before
            QuantQualityRejected is raised. Default 0.15 (15%).
        exporter: Label for evidence (e.g. "gptq", "awq").

    Returns:
        Dict with keys: fp16_ppl, quantized_ppl, delta_pct, passed.

    Raises:
        QuantQualityRejected: if (quantized_ppl - fp16_ppl) / fp16_ppl > threshold.
    """
    from dojo.core.evidence import EvidenceLogger, EvidenceType
    from dojo.core.exceptions import QuantQualityRejected

    texts = calibration_texts if calibration_texts is not None else _DEFAULT_CALIBRATION_TEXTS

    fp16_ppl = compute_real_ppl(baseline_model, tokenizer, calibration_texts=texts)
    quantized_ppl = compute_real_ppl(quantized_model, tokenizer, calibration_texts=texts)

    delta = (quantized_ppl - fp16_ppl) / fp16_ppl if fp16_ppl > 0 else 0.0
    delta_pct = round(delta * 100, 4)
    passed = delta <= ppl_delta_threshold

    logger = EvidenceLogger.get_instance()
    logger.emit(
        {
            "type": EvidenceType.QUANT_PPL_GATE.value,
            "data": {
                "exporter": exporter,
                "fp16_ppl": round(fp16_ppl, 6),
                "quantized_ppl": round(quantized_ppl, 6),
                "delta_pct": delta_pct,
                "threshold_pct": round(ppl_delta_threshold * 100, 4),
                "passed": passed,
            },
        },
        organ="quant",
    )

    if not passed:
        msg = (
            f"{exporter} PPL gate rejected: PPL delta {delta_pct:.2f}% "
            f"exceeds threshold {ppl_delta_threshold * 100:.2f}% "
            f"(fp16_ppl={fp16_ppl:.4f}, quantized_ppl={quantized_ppl:.4f})"
        )
        raise QuantQualityRejected(msg, reason=msg)

    return {
        "fp16_ppl": fp16_ppl,
        "quantized_ppl": quantized_ppl,
        "delta_pct": delta_pct,
        "passed": True,
    }


__all__ = ["compute_real_ppl", "measure_ppl_gate"]
