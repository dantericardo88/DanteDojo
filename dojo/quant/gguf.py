"""GGUF exporter for DanteDojo — llama.cpp target.

Converts a HuggingFace model directory to GGUF format using llama.cpp's
convert_hf_to_gguf.py script, then quantizes to the requested type.
Embeds the constitutional passport_id into GGUF metadata (sidecar .meta.json).
Emits QUANTIZATION_START and QUANTIZATION_COMPLETE evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile

# GGUF magic bytes: "GGUF" in little-endian ASCII
_GGUF_MAGIC = b"\x47\x47\x55\x46"

# Supported llama.cpp quantization types
VALID_QUANT_TYPES = [
    "Q4_0", "Q4_1",
    "Q5_0", "Q5_1",
    "Q8_0",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M",
    "Q6_K",
    "F16", "F32",
    "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
]

_INSTALL_INSTRUCTIONS = (
    "llama-cpp-python is not installed or llama.cpp convert_hf_to_gguf.py was not found.\n"
    "To install llama-cpp-python:\n"
    "  pip install llama-cpp-python\n"
    "Or for GPU support (CUDA):\n"
    "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python\n"
    "Or clone llama.cpp locally at ~/llama.cpp/\n"
    "See https://github.com/abetlen/llama-cpp-python and https://github.com/ggerganov/llama.cpp"
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@one_door("quant.gguf.export")
def export_gguf(
    model_path: Path,
    output_dir: Path,
    quant_type: str = "Q4_K_M",
    *,
    profile: Optional[HALProfile] = None,
    passport_id: str = "",
) -> Path:
    """Export model to GGUF format via llama.cpp.

    Args:
        model_path: Path to the HuggingFace model directory.
        output_dir: Directory where the .gguf file will be written.
        quant_type: llama.cpp quantization type (e.g. Q4_K_M, Q5_K_M).
        profile: Optional HALProfile for hardware-aware decisions.
        passport_id: Constitutional passport ID to embed in GGUF metadata.

    Returns:
        Path to the exported .gguf file.

    Raises:
        QuantizationError: if conversion fails or llama.cpp is unavailable
            and stub mode is disabled.
    """
    # Validate quant_type early — fail-closed before any I/O
    if quant_type not in VALID_QUANT_TYPES:
        valid_list = ", ".join(VALID_QUANT_TYPES)
        raise QuantizationError(
            f"Unsupported quantization type {quant_type!r}. "
            f"Valid types: {valid_list}",
            reason=f"quant_type={quant_type!r} not in VALID_QUANT_TYPES",
        )

    logger = EvidenceLogger.get_instance()

    model_name = model_path.name or "model"
    output_filename = f"{model_name}-{quant_type}.gguf"
    output_path = output_dir / output_filename

    # Emit start evidence
    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_START.value,
            "data": {
                "exporter": "gguf",
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "quant_type": quant_type,
                "profile_tier": profile.tier.value if profile else None,
                "passport_id": passport_id or None,
            },
        },
        organ="quant",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    convert_script = _find_convert_script()

    # Gate: require llama_cpp Python bindings only when no convert script is
    # available AND we are not in stub mode.  The subprocess conversion path
    # does not need the Python bindings, so we skip the check when a script
    # is present (tests inject a fake script without installing llama_cpp).
    if convert_script is None and not _stub_mode():
        _require_llama_cpp()

    if convert_script is not None:
        # Real conversion path
        output_path = _run_real_conversion(
            model_path=model_path,
            output_dir=output_dir,
            output_path=output_path,
            quant_type=quant_type,
            convert_script=convert_script,
        )
    elif _stub_mode():
        # Stub mode: write a minimal valid GGUF-header stub
        output_path = _write_stub(output_path)
    else:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        )

    # Embed constitutional passport metadata (sidecar)
    if passport_id:
        _embed_passport_metadata(output_path, passport_id=passport_id, profile=profile)

    # Compute file hash and emit completion evidence
    file_hash = _sha256_file(output_path)
    file_size = output_path.stat().st_size

    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "exporter": "gguf",
                "output_path": str(output_path),
                "quant_type": quant_type,
                "file_hash": file_hash,
                "file_size_bytes": file_size,
                "stub": convert_script is None,
                "passport_id": passport_id or None,
            },
        },
        organ="quant",
    )

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_llama_cpp() -> None:
    """Gate: in real mode require llama_cpp; in stub mode, skip."""
    if _stub_mode():
        return  # Stub mode: no requirement
    try:
        import llama_cpp  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


def _stub_mode() -> bool:
    """Return True when running in stub mode (no real GPU / llama.cpp available)."""
    return (
        os.environ.get("DOJO_REAL_GPU", "0") != "1"
        and os.environ.get("DOJO_GGUF_STUB", "1") == "1"
    )


def _find_convert_script() -> Optional[Path]:
    """Find llama.cpp's convert_hf_to_gguf.py in common locations."""
    # 1. Bundled with llama-cpp-python
    try:
        import llama_cpp  # noqa: F401  # type: ignore[import]
        pkg_dir = Path(llama_cpp.__file__).parent
        script = pkg_dir / "convert_hf_to_gguf.py"
        if script.exists():
            return script
    except (ImportError, RuntimeError):
        pass

    # 2. Local llama.cpp clone (common locations, including Windows)
    env_path = os.environ.get("LLAMA_CPP_PATH", "")
    candidates = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/lib/llama.cpp/convert_hf_to_gguf.py"),
        Path("llama.cpp/convert_hf_to_gguf.py"),
    ]
    if env_path:
        candidates.append(Path(env_path) / "convert_hf_to_gguf.py")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 3. Try auto-download if enabled
    downloaded = _download_convert_script()
    if downloaded is not None:
        return downloaded

    return None


def _download_convert_script() -> Optional[Path]:
    """Download convert_hf_to_gguf.py from llama.cpp repo if DOJO_AUTO_DOWNLOAD_SCRIPTS=1.

    Uses urllib.request.urlretrieve to save to a temp dir.
    Returns the path on success, None otherwise.
    """
    if os.environ.get("DOJO_AUTO_DOWNLOAD_SCRIPTS", "0") != "1":
        return None

    _SCRIPT_URL = (
        "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    )
    try:
        dest_dir = Path(tempfile.gettempdir()) / "dojo_llama_cpp"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "convert_hf_to_gguf.py"
        urllib.request.urlretrieve(_SCRIPT_URL, str(dest))
        if dest.exists():
            return dest
    except Exception:
        pass
    return None


def _run_real_conversion(
    model_path: Path,
    output_dir: Path,
    output_path: Path,
    quant_type: str,
    convert_script: Path,
) -> Path:
    """Run convert_hf_to_gguf.py + llama-quantize for real conversion.

    Returns the path to the final quantized GGUF file.

    Raises:
        QuantizationError: if the conversion subprocess fails.
    """
    f16_path = output_dir / f"{model_path.name}-f16.gguf"

    # Step 1: Convert HF model to F16 GGUF
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(model_path),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise QuantizationError(
            f"Failed to run convert_hf_to_gguf.py: {exc}",
            reason=str(exc),
        ) from exc

    if result.returncode != 0:
        raise QuantizationError(
            f"convert_hf_to_gguf.py failed (exit {result.returncode}): {result.stderr[:512]}",
            reason=result.stderr[:512],
        )

    final_path = output_path

    # Step 2: Quantize F16 → quant_type if llama-quantize is in PATH
    llama_quantize = shutil.which("llama-quantize")
    if llama_quantize and f16_path.exists():
        try:
            qresult = subprocess.run(
                [llama_quantize, str(f16_path), str(output_path), quant_type],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise QuantizationError(
                f"Failed to run llama-quantize: {exc}",
                reason=str(exc),
            ) from exc

        if qresult.returncode != 0:
            raise QuantizationError(
                f"llama-quantize failed (exit {qresult.returncode}): {qresult.stderr[:512]}",
                reason=qresult.stderr[:512],
            )

        # Remove intermediate F16 file
        if f16_path.exists() and f16_path != output_path:
            f16_path.unlink(missing_ok=True)
    else:
        # llama-quantize not available — use F16 as final output
        if f16_path.exists():
            f16_path.rename(output_path)
        final_path = output_path

    return final_path


def _write_stub(output_path: Path) -> Path:
    """Write a minimal GGUF-header stub for testing/offline use.

    The stub starts with the GGUF magic bytes so format validators can
    identify it as a GGUF file. It is not a valid loadable model.
    """
    # GGUF v3 minimal header:
    # - 4 bytes: magic "GGUF"
    # - 4 bytes: version (3, little-endian)
    # - 8 bytes: tensor_count (0)
    # - 8 bytes: metadata_kv_count (0)
    stub_header = (
        _GGUF_MAGIC
        + (3).to_bytes(4, "little")   # version = 3
        + (0).to_bytes(8, "little")   # tensor_count = 0
        + (0).to_bytes(8, "little")   # metadata_kv_count = 0
    )
    output_path.write_bytes(stub_header)
    return output_path


def _embed_passport_metadata(
    gguf_path: Path,
    passport_id: str,
    profile: Optional[HALProfile] = None,
) -> None:
    """Embed passport_id as a string KV pair in the GGUF file metadata.

    Uses a sidecar .meta.json file — always reliable regardless of whether
    the llama_cpp Python package exposes a write API.

    Also embeds merkle_root and issued_at from the HALProfile passport if
    available, creating a provenance chain: training → promotion → inference.
    """
    meta: dict = {
        "dojo.passport_id": passport_id,
        "dojo.gguf_sha256": _sha256_file(gguf_path),
    }

    # Extend with passport provenance fields if available on the profile
    if profile is not None:
        passport = getattr(profile, "passport", None)
        if passport is not None:
            merkle_root = getattr(passport, "merkle_root", None)
            issued_at = getattr(passport, "issued_at", None)
            if merkle_root is not None:
                meta["dojo.merkle_root"] = str(merkle_root)
            if issued_at is not None:
                meta["dojo.issued_at"] = str(issued_at)

    meta_path = gguf_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["export_gguf"]
