"""DanteDojo kernels package — constitutional custom CUDA/Triton kernels.

All kernel access MUST go through the one-door dispatch functions in
``dojo.kernels.dispatch``.  Direct use of ``attention.py`` or ``rms_norm.py``
is permitted only from ``dispatch.py`` and tests.

Exposed API:
- ``flash_attention_forward``  — attention kernel (Triton or PyTorch).
- ``fused_rms_norm``           — RMS norm forward kernel.
- ``fused_rms_norm_backward``  — RMS norm backward kernel.
- ``is_triton_available``      — runtime Triton availability check.
- ``KernelRegistry``           — probe and select best available kernels.
- ``dispatch_attention``       — one-door attention dispatch.
- ``dispatch_rms_norm``        — one-door RMS norm dispatch.
"""

from dojo.kernels.attention import flash_attention_forward, is_triton_available
from dojo.kernels.dispatch import dispatch_attention, dispatch_rms_norm
from dojo.kernels.registry import KernelRegistry
from dojo.kernels.rms_norm import fused_rms_norm, fused_rms_norm_backward

__all__ = [
    "flash_attention_forward",
    "is_triton_available",
    "fused_rms_norm",
    "fused_rms_norm_backward",
    "KernelRegistry",
    "dispatch_attention",
    "dispatch_rms_norm",
]
