"""
MPS Flash Attention - Flash Attention for PyTorch on Apple Silicon

This package provides memory-efficient attention using Metal Flash Attention kernels.
"""

import torch
from typing import Optional
import math
import threading
import os

# Try to import the C++ extension
try:
    from . import _C
    _HAS_MFA = True
except ImportError as e:
    _HAS_MFA = False
    _IMPORT_ERROR = str(e)

# Set up shipped kernels directory for zero-compilation loading
def _init_shipped_kernels():
    """Point the Swift bridge to pre-shipped kernel binaries."""
    try:
        import ctypes
        bridge_path = os.environ.get("MFA_BRIDGE_PATH")
        if not bridge_path:
            module_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(module_dir, "lib", "libMFABridge.dylib"),  # Bundled in wheel
                os.path.join(module_dir, "..", "swift-bridge", ".build", "release", "libMFABridge.dylib"),
                os.path.join(module_dir, "libMFABridge.dylib"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    bridge_path = path
                    break

        if bridge_path and os.path.exists(bridge_path):
            lib = ctypes.CDLL(bridge_path)

            # Set shipped kernels directory (pre-compiled metallibs + pipeline binaries)
            kernels_dir = os.path.join(os.path.dirname(__file__), "kernels")
            if os.path.exists(kernels_dir):
                lib.mfa_set_kernels_dir(kernels_dir.encode('utf-8'))

            lib.mfa_init()
    except Exception:
        pass  # Init is optional, will fall back to runtime compilation

# Initialize shipped kernels on import
if _HAS_MFA:
    _init_shipped_kernels()


def is_available() -> bool:
    """Check if MPS Flash Attention is available."""
    return _HAS_MFA and torch.backends.mps.is_available()


class FlashAttentionFunction(torch.autograd.Function):
    """Autograd function for Flash Attention with backward pass support."""

    @staticmethod
    def forward(ctx, query, key, value, is_causal, scale):
        # Apply scale if provided (MFA uses 1/sqrt(D) internally)
        scale_factor = 1.0
        if scale is not None:
            default_scale = 1.0 / math.sqrt(query.shape[-1])
            if abs(scale - default_scale) > 1e-6:
                scale_factor = scale / default_scale
                query = query * scale_factor

        # Forward with logsumexp for backward
        output, logsumexp = _C.forward_with_lse(query, key, value, is_causal)

        # Save for backward
        ctx.save_for_backward(query, key, value, output, logsumexp)
        ctx.is_causal = is_causal
        ctx.scale_factor = scale_factor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, logsumexp = ctx.saved_tensors

        # Compute gradients
        dQ, dK, dV = _C.backward(
            grad_output, query, key, value, output, logsumexp, ctx.is_causal
        )

        # If we scaled the query in forward, scale the gradient back
        if ctx.scale_factor != 1.0:
            dQ = dQ * ctx.scale_factor

        # Return gradients (None for is_causal and scale since they're not tensors)
        return dQ, dK, dV, None, None


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention using Flash Attention on MPS.

    This function provides O(N) memory complexity instead of O(N²) by using
    tiled computation, allowing much longer sequences on limited GPU memory.

    Supports both forward and backward passes for training.

    Args:
        query: Query tensor of shape (B, num_heads, seq_len, head_dim)
        key: Key tensor of shape (B, num_heads, seq_len, head_dim)
        value: Value tensor of shape (B, num_heads, seq_len, head_dim)
        is_causal: If True, applies causal masking (for autoregressive models)
        scale: Scaling factor for attention scores. Default: 1/sqrt(head_dim)

    Returns:
        Output tensor of shape (B, num_heads, seq_len, head_dim)

    Example:
        >>> import torch
        >>> from mps_flash_attn import flash_attention
        >>> q = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
        >>> out = flash_attention(q, k, v)

        # With gradients:
        >>> q.requires_grad = True
        >>> out = flash_attention(q, k, v)
        >>> out.sum().backward()  # Computes dQ
    """
    if not _HAS_MFA:
        raise RuntimeError(
            f"MPS Flash Attention C++ extension not available: {_IMPORT_ERROR}\n"
            "Please rebuild with: pip install -e ."
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")

    # Validate device
    if query.device.type != 'mps':
        raise ValueError("query must be on MPS device")
    if key.device.type != 'mps':
        raise ValueError("key must be on MPS device")
    if value.device.type != 'mps':
        raise ValueError("value must be on MPS device")

    # Use autograd function for gradient support
    return FlashAttentionFunction.apply(query, key, value, is_causal, scale)


def replace_sdpa():
    """
    Monkey-patch torch.nn.functional.scaled_dot_product_attention to use
    Flash Attention on MPS devices.

    Call this at the start of your script to automatically use Flash Attention
    for all attention operations.
    """
    import torch.nn.functional as F

    original_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None):
        # Use MFA for MPS tensors without mask/dropout
        if (query.device.type == 'mps' and
            attn_mask is None and
            dropout_p == 0.0 and
            _HAS_MFA):
            try:
                return flash_attention(query, key, value, is_causal=is_causal, scale=scale)
            except Exception:
                # Fall back to original on any error
                pass

        return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)

    F.scaled_dot_product_attention = patched_sdpa
    print("MPS Flash Attention: Patched F.scaled_dot_product_attention")


def precompile():
    """
    Pre-compile Metal kernels for common configurations.

    Call this once after installation to eliminate runtime compilation overhead.
    Pre-compiled kernels are cached to disk and loaded instantly on subsequent runs.

    This compiles kernels for:
    - Sequence lengths: 64, 128, 256, 512, 1024, 2048, 4096, 8192
    - Head dimensions: 32, 48, 64, 80, 96, 128
    - Both fp32 and fp16 precision

    Total: 96 kernel configurations
    """
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    import ctypes
    import os

    # Load the Swift bridge directly
    bridge_path = os.environ.get("MFA_BRIDGE_PATH")
    if not bridge_path:
        # Try common locations
        module_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(module_dir, "lib", "libMFABridge.dylib"),  # Bundled in wheel
            os.path.join(module_dir, "..", "swift-bridge", ".build", "release", "libMFABridge.dylib"),
            os.path.join(module_dir, "libMFABridge.dylib"),
        ]
        for path in candidates:
            if os.path.exists(path):
                bridge_path = path
                break

    if not bridge_path or not os.path.exists(bridge_path):
        raise RuntimeError("Cannot find libMFABridge.dylib. Set MFA_BRIDGE_PATH environment variable.")

    lib = ctypes.CDLL(bridge_path)
    lib.mfa_precompile()
    print("\nPre-compilation complete! Kernels cached to disk.")


def clear_cache():
    """Clear the pre-compiled kernel cache."""
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    import ctypes
    import os

    bridge_path = os.environ.get("MFA_BRIDGE_PATH")
    if bridge_path and os.path.exists(bridge_path):
        lib = ctypes.CDLL(bridge_path)
        lib.mfa_clear_cache()
        print("Cache cleared.")
