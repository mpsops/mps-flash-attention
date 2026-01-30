"""
MPS Flash Attention - Flash Attention for PyTorch on Apple Silicon

This package provides memory-efficient attention using Metal Flash Attention kernels.
"""

__version__ = "0.2.1"

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
    def forward(ctx, query, key, value, is_causal, scale, attn_mask, window_size, bf16_backward):
        # Apply scale if provided (MFA uses 1/sqrt(D) internally)
        scale_factor = 1.0
        if scale is not None:
            default_scale = 1.0 / math.sqrt(query.shape[-1])
            if abs(scale - default_scale) > 1e-6:
                scale_factor = scale / default_scale
                query = query * scale_factor

        # Forward with logsumexp for backward
        output, logsumexp = _C.forward_with_lse(query, key, value, is_causal, attn_mask, window_size)

        # Save for backward
        if attn_mask is not None:
            ctx.save_for_backward(query, key, value, output, logsumexp, attn_mask)
            ctx.has_mask = True
        else:
            ctx.save_for_backward(query, key, value, output, logsumexp)
            ctx.has_mask = False
        ctx.is_causal = is_causal
        ctx.scale_factor = scale_factor
        ctx.window_size = window_size
        ctx.bf16_backward = bf16_backward

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_mask:
            query, key, value, output, logsumexp, attn_mask = ctx.saved_tensors
        else:
            query, key, value, output, logsumexp = ctx.saved_tensors
            attn_mask = None

        # Compute gradients with optional BF16 mixed-precision
        dQ, dK, dV = _C.backward(
            grad_output, query, key, value, output, logsumexp, ctx.is_causal, attn_mask, ctx.window_size, ctx.bf16_backward
        )

        # If we scaled the query in forward, scale the gradient back
        if ctx.scale_factor != 1.0:
            dQ = dQ * ctx.scale_factor

        # Return gradients (None for non-tensor args that don't need grad)
        return dQ, dK, dV, None, None, None, None, None


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    window_size: int = 0,
    bf16_backward: bool = False,
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
        window_size: Sliding window attention size. If 0 (default), uses full attention.
                     If > 0, each token only attends to the previous window_size tokens.
                     Used by models like Mistral and Llama 3.2 for efficient long context.
        bf16_backward: If True, use BF16 for backward pass intermediates. This provides
                       ~2x speedup on backward pass with minimal accuracy loss (<1%).
                       Recommended for training large models. Default: False.

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

        # Fast training with BF16 backward:
        >>> out = flash_attention(q, k, v, bf16_backward=True)
        >>> out.sum().backward()  # ~2x faster backward

        # With attention mask:
        >>> mask = torch.zeros(2, 1, 4096, 4096, dtype=torch.bool, device='mps')
        >>> mask[:, :, :, 2048:] = True  # mask out second half of keys
        >>> out = flash_attention(q, k, v, attn_mask=mask)

        # With sliding window (Mistral-style):
        >>> out = flash_attention(q, k, v, is_causal=True, window_size=4096)
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
        return _C.forward(query, key, value, is_causal, attn_mask, window_size)

    # Use autograd function for gradient support
    return FlashAttentionFunction.apply(query, key, value, is_causal, scale, attn_mask, window_size, bf16_backward)


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
                     is_causal=False, scale=None, enable_gqa=False, **kwargs):
        # Use MFA for MPS tensors without dropout
        # Only use MFA for seq_len >= 1024 where it outperforms PyTorch's math backend
        # For shorter sequences, PyTorch's simpler matmul+softmax approach is faster
        # Benchmark results (B=1-4, H=8, D=64-128, fp16/bf16):
        #   seq=512:  0.3-0.5x (MFA slower)
        #   seq=1024: 1.1-2.0x (MFA faster)
        #   seq=2048: 1.7-3.7x (MFA much faster)
        #   seq=4096: 2.0-3.9x (MFA much faster)
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

        return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa, **kwargs)

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


def flash_attention_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 16384,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention for very long sequences using chunked computation.

    This function enables processing of 100K+ token sequences without OOM by:
    1. Processing K/V in chunks
    2. Using online softmax correction to maintain numerical accuracy
    3. Fusing chunk results incrementally

    The memory complexity is O(chunk_size) instead of O(seq_len) for the K/V cache.

    Args:
        query: Query tensor of shape (B, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (B, num_heads, seq_len_kv, head_dim)
        value: Value tensor of shape (B, num_heads, seq_len_kv, head_dim)
        chunk_size: Size of K/V chunks to process. Default: 16384.
                    Larger = faster but more memory. Smaller = slower but less memory.
        is_causal: If True, applies causal masking.
        scale: Scaling factor for attention scores. Default: 1/sqrt(head_dim)

    Returns:
        Output tensor of shape (B, num_heads, seq_len_q, head_dim)

    Example:
        >>> import torch
        >>> from mps_flash_attn import flash_attention_chunked
        >>> # Process 100K sequence
        >>> q = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
        >>> k = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
        >>> v = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
        >>> out = flash_attention_chunked(q, k, v, chunk_size=16384)

    Note:
        - This function does NOT support backward pass (use for inference only)
        - For training with long sequences, consider gradient checkpointing
        - Performance is best when chunk_size is a multiple of 64
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

    B, H, seq_len_q, D = query.shape
    _, _, seq_len_kv, _ = key.shape

    # Apply scale to query if provided
    if scale is not None:
        default_scale = 1.0 / math.sqrt(D)
        if abs(scale - default_scale) > 1e-6:
            scale_factor = scale / default_scale
            query = query * scale_factor

    # If sequence fits in one chunk, use regular attention
    if seq_len_kv <= chunk_size:
        return _C.forward(query, key, value, is_causal, None, 0)

    # Initialize running statistics for online softmax
    # m = running max, l = running sum of exp, acc = accumulated output
    device = query.device
    dtype = query.dtype

    # Use float32 for numerical stability of softmax statistics
    running_max = torch.full((B, H, seq_len_q, 1), float('-inf'), device=device, dtype=torch.float32)
    running_sum = torch.zeros((B, H, seq_len_q, 1), device=device, dtype=torch.float32)
    output_acc = torch.zeros((B, H, seq_len_q, D), device=device, dtype=torch.float32)

    # Process K/V in chunks
    num_chunks = (seq_len_kv + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, seq_len_kv)

        # Extract chunk
        k_chunk = key[:, :, start_idx:end_idx, :]
        v_chunk = value[:, :, start_idx:end_idx, :]

        # For causal attention, we need to handle the mask properly
        # Each query position q can only attend to k positions where k <= q
        # For chunk [start_idx, end_idx), a query at position q attends to:
        # - All of chunk if q >= end_idx
        # - Partial chunk (up to q) if start_idx <= q < end_idx
        # - None of chunk if q < start_idx

        chunk_is_causal = is_causal and (end_idx <= seq_len_q)

        # Compute attention for this chunk
        # forward_with_lse returns (output, logsumexp) where logsumexp = m + log(l)
        chunk_out, chunk_lse = _C.forward_with_lse(query, k_chunk, v_chunk, chunk_is_causal, None, 0)

        # chunk_lse shape: (B, H, seq_len_q)
        # We need to convert logsumexp to (max, sum) for online algorithm
        chunk_lse = chunk_lse.unsqueeze(-1)  # (B, H, seq_len_q, 1)

        # Convert chunk output to float32 for accumulation
        chunk_out = chunk_out.float()

        # Online softmax update:
        # new_max = max(running_max, chunk_max)
        # For flash attention, chunk_lse ≈ chunk_max + log(chunk_sum)
        # We approximate chunk_max ≈ chunk_lse (valid when exp sum dominates)

        chunk_max = chunk_lse  # Approximation: logsumexp ≈ max when sum is dominated by max

        # Compute new max
        new_max = torch.maximum(running_max, chunk_max)

        # Rescale previous accumulator
        # correction_old = exp(running_max - new_max)
        correction_old = torch.exp(running_max - new_max)
        # Clip to avoid inf * 0 issues when running_max was -inf
        correction_old = torch.where(running_max == float('-inf'), torch.zeros_like(correction_old), correction_old)

        # Rescale chunk output
        # correction_new = exp(chunk_max - new_max)
        correction_new = torch.exp(chunk_max - new_max)

        # For the sum, we need exp(chunk_lse - new_max) = exp(chunk_max + log(chunk_sum) - new_max)
        # = exp(chunk_max - new_max) * chunk_sum
        # But we only have logsumexp, so: exp(chunk_lse - new_max)
        chunk_sum_scaled = torch.exp(chunk_lse - new_max)

        # Update accumulator
        output_acc = output_acc * correction_old + chunk_out * correction_new
        running_sum = running_sum * correction_old + chunk_sum_scaled
        running_max = new_max

    # Final normalization
    output = output_acc / running_sum

    # Convert back to original dtype
    return output.to(dtype)


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


# =============================================================================
# Quantized Attention (FP8, INT8, NF4)
# =============================================================================

# Quantization type constants
QUANT_FP8_E4M3 = 3  # FP8 with 4 exponent bits, 3 mantissa bits (better precision)
QUANT_FP8_E5M2 = 4  # FP8 with 5 exponent bits, 2 mantissa bits (better range)
QUANT_INT8 = 5      # INT8 with per-head scaling
QUANT_NF4 = 6       # NormalFloat 4-bit (for 4-bit quantization)


def quantize_kv_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    use_e5m2: bool = False,
) -> tuple:
    """
    Quantize Key and Value tensors to FP8 format.

    FP8 quantization provides 2x memory reduction with minimal accuracy loss.
    Two formats are available:
    - E4M3 (default): 4 exponent bits, 3 mantissa bits - better precision
    - E5M2: 5 exponent bits, 2 mantissa bits - better dynamic range

    Args:
        key: Key tensor of shape (B, H, N, D)
        value: Value tensor of shape (B, H, N, D)
        use_e5m2: If True, use E5M2 format. Default: False (E4M3)

    Returns:
        Tuple of (key_quant, value_quant, k_scale, v_scale) where:
        - key_quant, value_quant: uint8 tensors with quantized values
        - k_scale, v_scale: float32 tensors with per-head scale factors

    Example:
        >>> k_q, v_q, k_s, v_s = quantize_kv_fp8(key, value)
        >>> out = flash_attention_fp8(query, k_q, v_q, k_s, v_s)
    """
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    k_quant, k_scale = _C.quantize_to_fp8(key, use_e5m2)
    v_quant, v_scale = _C.quantize_to_fp8(value, use_e5m2)
    return k_quant, v_quant, k_scale, v_scale


def quantize_kv_int8(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Quantize Key and Value tensors to INT8 format.

    INT8 quantization provides 2x memory reduction with symmetric quantization
    and per-head scaling.

    Args:
        key: Key tensor of shape (B, H, N, D)
        value: Value tensor of shape (B, H, N, D)

    Returns:
        Tuple of (key_quant, value_quant, k_scale, v_scale) where:
        - key_quant, value_quant: uint8 tensors (centered at 128)
        - k_scale, v_scale: float32 tensors with per-head scale factors

    Example:
        >>> k_q, v_q, k_s, v_s = quantize_kv_int8(key, value)
        >>> out = flash_attention_int8(query, k_q, v_q, k_s, v_s)
    """
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    k_quant, k_scale = _C.quantize_to_int8(key)
    v_quant, v_scale = _C.quantize_to_int8(value)
    return k_quant, v_quant, k_scale, v_scale


def flash_attention_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    window_size: int = 0,
    use_e5m2: bool = False,
) -> torch.Tensor:
    """
    Compute attention with FP8 quantized Key/Value tensors.

    This function provides 2x memory reduction for K/V cache with minimal
    accuracy impact. Useful for KV cache compression in long-context inference.

    Args:
        query: Query tensor (B, H, N, D) in FP16/BF16/FP32
        key: Quantized Key tensor (B, H, N, D) as uint8
        value: Quantized Value tensor (B, H, N, D) as uint8
        k_scale: Per-head scale for K (B, H) or (H,)
        v_scale: Per-head scale for V (B, H) or (H,)
        is_causal: If True, applies causal masking
        attn_mask: Optional boolean attention mask
        window_size: Sliding window size (0 = full attention)
        use_e5m2: If True, use E5M2 format. Default: False (E4M3)

    Returns:
        Output tensor of shape (B, H, N, D)

    Example:
        >>> # First quantize K/V
        >>> k_q, v_q, k_s, v_s = quantize_kv_fp8(key, value)
        >>> # Then compute attention
        >>> out = flash_attention_fp8(query, k_q, v_q, k_s, v_s)
    """
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    quant_type = QUANT_FP8_E5M2 if use_e5m2 else QUANT_FP8_E4M3
    return _C.forward_quantized(
        query, key, value, k_scale, v_scale,
        quant_type, is_causal, attn_mask, window_size
    )


def flash_attention_int8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    window_size: int = 0,
) -> torch.Tensor:
    """
    Compute attention with INT8 quantized Key/Value tensors.

    This function provides 2x memory reduction for K/V cache using symmetric
    INT8 quantization with per-head scaling.

    Args:
        query: Query tensor (B, H, N, D) in FP16/BF16/FP32
        key: Quantized Key tensor (B, H, N, D) as uint8
        value: Quantized Value tensor (B, H, N, D) as uint8
        k_scale: Per-head scale for K (B, H) or (H,)
        v_scale: Per-head scale for V (B, H) or (H,)
        is_causal: If True, applies causal masking
        attn_mask: Optional boolean attention mask
        window_size: Sliding window size (0 = full attention)

    Returns:
        Output tensor of shape (B, H, N, D)

    Example:
        >>> k_q, v_q, k_s, v_s = quantize_kv_int8(key, value)
        >>> out = flash_attention_int8(query, k_q, v_q, k_s, v_s)
    """
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    return _C.forward_quantized(
        query, key, value, k_scale, v_scale,
        QUANT_INT8, is_causal, attn_mask, window_size
    )


def flash_attention_quantized(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    quant_type: int,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    window_size: int = 0,
) -> torch.Tensor:
    """
    Generic quantized attention with configurable quantization type.

    Low-level function that accepts any supported quantization type.
    For convenience, prefer using flash_attention_fp8() or flash_attention_int8().

    Args:
        query: Query tensor (B, H, N, D) in FP16/BF16/FP32
        key: Quantized Key tensor (B, H, N, D) as uint8
        value: Quantized Value tensor (B, H, N, D) as uint8
        k_scale: Per-head scale for K
        v_scale: Per-head scale for V
        quant_type: Quantization type constant:
            - QUANT_FP8_E4M3 (3): FP8 with E4M3 format
            - QUANT_FP8_E5M2 (4): FP8 with E5M2 format
            - QUANT_INT8 (5): INT8 with symmetric quantization
            - QUANT_NF4 (6): NormalFloat 4-bit (experimental)
        is_causal: If True, applies causal masking
        attn_mask: Optional boolean attention mask
        window_size: Sliding window size (0 = full attention)

    Returns:
        Output tensor of shape (B, H, N, D)
    """
    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    return _C.forward_quantized(
        query, key, value, k_scale, v_scale,
        quant_type, is_causal, attn_mask, window_size
    )


# Lazy import benchmark module to avoid circular imports
_benchmark_module = None

def __getattr__(name):
    global _benchmark_module
    if name == "benchmark":
        if _benchmark_module is None:
            # Use importlib to avoid recursion from "from mps_flash_attn import benchmark"
            import importlib
            _benchmark_module = importlib.import_module(".benchmark", __name__)
        return _benchmark_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# PyTorch Custom Op Registration for torch.compile() support
# =============================================================================

# Module-level storage to prevent garbage collection of registered Library
_mfa_lib = None


def _register_custom_op():
    """
    Register MFA as a PyTorch custom op for torch.compile() compatibility.

    This allows MFA to work seamlessly with torch.compile() and other
    PyTorch JIT infrastructure.

    Usage:
        import torch
        from mps_flash_attn import flash_attention, register_custom_op

        register_custom_op()

        @torch.compile
        def my_attention(q, k, v):
            return torch.ops.mfa.flash_attention(q, k, v)

        # Works with compiled models!
        out = my_attention(q, k, v)
    """
    global _mfa_lib

    if not _HAS_MFA:
        raise RuntimeError(f"MPS Flash Attention not available: {_IMPORT_ERROR}")

    # Already registered
    if _mfa_lib is not None:
        return True

    try:
        from torch.library import Library, impl

        # Create the library and store in module-level variable to prevent GC
        _mfa_lib = Library("mfa", "DEF")

        # Define the operation schema
        _mfa_lib.define(
            "flash_attention(Tensor query, Tensor key, Tensor value, "
            "bool is_causal=False, Tensor? attn_mask=None, int window_size=0) -> Tensor"
        )

        # MPS implementation
        @impl(_mfa_lib, "flash_attention", "MPS")
        def flash_attention_mps(query, key, value, is_causal=False, attn_mask=None, window_size=0):
            return flash_attention(query, key, value, is_causal=is_causal, attn_mask=attn_mask, window_size=window_size)

        # Meta implementation for shape inference during tracing
        @impl(_mfa_lib, "flash_attention", "Meta")
        def flash_attention_meta(query, key, value, is_causal=False, attn_mask=None, window_size=0):
            # Return a tensor with the same shape as query (output shape matches query)
            return query.new_empty(query.shape)

        # CPU fallback (uses PyTorch's SDPA)
        @impl(_mfa_lib, "flash_attention", "CPU")
        def flash_attention_cpu(query, key, value, is_causal=False, attn_mask=None, window_size=0):
            import torch.nn.functional as F
            # Note: PyTorch SDPA doesn't support sliding window, so we ignore window_size on CPU
            if window_size > 0:
                import warnings
                warnings.warn("Sliding window attention not supported on CPU, using full attention")
            return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=is_causal)

        print("MFA custom op registered: torch.ops.mfa.flash_attention")
        return True

    except ImportError:
        print("Warning: torch.library not available, custom op registration skipped")
        return False
    except Exception as e:
        print(f"Warning: Failed to register custom op: {e}")
        return False


def register_custom_op():
    """
    Register MFA as a PyTorch custom op for torch.compile() support.

    Call this once at module import time if you want to use torch.compile()
    with MFA. After registration, you can use:

        torch.ops.mfa.flash_attention(q, k, v, is_causal=False, attn_mask=None, window_size=0)

    This is compatible with torch.compile() and torch.jit.trace().
    """
    return _register_custom_op()


# Auto-register custom op if TORCH_COMPILE is set
if os.environ.get("MFA_REGISTER_CUSTOM_OP", "0") == "1":
    _register_custom_op()
