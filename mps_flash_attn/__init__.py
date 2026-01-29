"""
MPS Flash Attention - Flash Attention for PyTorch on Apple Silicon

This package provides memory-efficient attention using Metal Flash Attention kernels.
"""

__version__ = "0.1.12"

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

# Note: The C++ extension handles loading libMFABridge.dylib via dlopen.
# Set MFA_BRIDGE_PATH environment variable to specify the library location.
# Do NOT load the library here via ctypes - that causes duplicate class warnings.


def is_available() -> bool:
    """Check if MPS Flash Attention is available."""
    return _HAS_MFA and torch.backends.mps.is_available()


def convert_mask(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Convert attention mask to MFA's boolean format.

    MFA uses boolean masks where True = masked (don't attend).
    PyTorch SDPA uses additive float masks where -inf/large negative = masked.

    Args:
        attn_mask: Optional mask, either:
            - None: no mask
            - bool tensor: already in MFA format (True = masked)
            - float tensor: additive mask (large negative = masked)

    Returns:
        Boolean mask suitable for flash_attention(), or None
    """
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        return attn_mask
    # Float mask: large negative values indicate masked positions
    return attn_mask <= -1e3


class FlashAttentionFunction(torch.autograd.Function):
    """Autograd function for Flash Attention with backward pass support."""

    @staticmethod
    def forward(ctx, query, key, value, is_causal, scale, attn_mask):
        # Apply scale if provided (MFA uses 1/sqrt(D) internally)
        scale_factor = 1.0
        if scale is not None:
            default_scale = 1.0 / math.sqrt(query.shape[-1])
            if abs(scale - default_scale) > 1e-6:
                scale_factor = scale / default_scale
                query = query * scale_factor

        # Forward with logsumexp for backward
        output, logsumexp = _C.forward_with_lse(query, key, value, is_causal, attn_mask)

        # Save for backward
        if attn_mask is not None:
            ctx.save_for_backward(query, key, value, output, logsumexp, attn_mask)
            ctx.has_mask = True
        else:
            ctx.save_for_backward(query, key, value, output, logsumexp)
            ctx.has_mask = False
        ctx.is_causal = is_causal
        ctx.scale_factor = scale_factor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_mask:
            query, key, value, output, logsumexp, attn_mask = ctx.saved_tensors
        else:
            query, key, value, output, logsumexp = ctx.saved_tensors
            attn_mask = None

        # Compute gradients
        dQ, dK, dV = _C.backward(
            grad_output, query, key, value, output, logsumexp, ctx.is_causal, attn_mask
        )

        # If we scaled the query in forward, scale the gradient back
        if ctx.scale_factor != 1.0:
            dQ = dQ * ctx.scale_factor

        # Return gradients (None for is_causal, scale, and attn_mask since they're not tensors or don't need grad)
        return dQ, dK, dV, None, None, None


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
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
        attn_mask: Optional boolean attention mask of shape (B, 1, seq_len_q, seq_len_kv)
                   or (B, num_heads, seq_len_q, seq_len_kv). True values indicate
                   positions to be masked (not attended to).

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

        # With attention mask:
        >>> mask = torch.zeros(2, 1, 4096, 4096, dtype=torch.bool, device='mps')
        >>> mask[:, :, :, 2048:] = True  # mask out second half of keys
        >>> out = flash_attention(q, k, v, attn_mask=mask)
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
    if attn_mask is not None and attn_mask.device.type != 'mps':
        raise ValueError("attn_mask must be on MPS device")

    # Fast path: inference mode (no grad) - skip autograd overhead and don't save tensors
    if not torch.is_grad_enabled() or (not query.requires_grad and not key.requires_grad and not value.requires_grad):
        # Apply scale if provided
        if scale is not None:
            default_scale = 1.0 / math.sqrt(query.shape[-1])
            if abs(scale - default_scale) > 1e-6:
                scale_factor = scale / default_scale
                query = query * scale_factor

        # Forward only - no logsumexp needed, no tensors saved
        return _C.forward(query, key, value, is_causal, attn_mask)

    # Use autograd function for gradient support
    return FlashAttentionFunction.apply(query, key, value, is_causal, scale, attn_mask)


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
        # Use MFA for MPS tensors without dropout
        # Only use MFA for seq_len >= 1024 where it outperforms PyTorch's math backend
        # For shorter sequences, PyTorch's simpler matmul+softmax approach is faster
        if (query.device.type == 'mps' and
            dropout_p == 0.0 and
            _HAS_MFA and
            query.shape[2] >= 1024):
            try:
                # Convert float mask to bool mask if needed
                # PyTorch SDPA uses additive masks (0 = attend, -inf = mask)
                # MFA uses boolean masks (False/0 = attend, True/non-zero = mask)
                mfa_mask = None
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        # Boolean mask: True means masked (don't attend)
                        mfa_mask = attn_mask
                    else:
                        # Float mask: typically -inf for masked positions, 0 for unmasked
                        # Convert: positions with large negative values -> True (masked)
                        # Use -1e3 threshold to catch -1000, -10000, -inf, etc.
                        mfa_mask = attn_mask <= -1e3
                return flash_attention(query, key, value, is_causal=is_causal, scale=scale, attn_mask=mfa_mask)
            except Exception:
                # Fall back to original on any error
                pass

        return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)

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
