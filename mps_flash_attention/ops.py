"""
Flash Attention operations for MPS.

This module provides the Python interface to the Metal Flash Attention kernels.
"""

import torch
from torch import Tensor
from typing import Optional
import math


class FlashAttentionFunction(torch.autograd.Function):
    """
    Autograd function for Flash Attention with custom backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: Optional[float] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass of Flash Attention.

        Args:
            q: Query tensor (B, H, N, D) or (B, N, H, D)
            k: Key tensor (B, H, N, D) or (B, N, H, D)
            v: Value tensor (B, H, N, D) or (B, N, H, D)
            scale: Attention scale (default: 1/sqrt(D))
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor with same shape as q
        """
        # Import the C extension
        try:
            from . import _C
        except ImportError:
            raise ImportError(
                "MPS Flash Attention C extension not found. "
                "Please rebuild with: pip install -e ."
            )

        # Validate inputs
        assert q.device.type == 'mps', "Query must be on MPS device"
        assert k.device.type == 'mps', "Key must be on MPS device"
        assert v.device.type == 'mps', "Value must be on MPS device"
        assert q.is_contiguous(), "Query must be contiguous"
        assert k.is_contiguous(), "Key must be contiguous"
        assert v.is_contiguous(), "Value must be contiguous"

        # Default scale
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])

        # Call Metal kernel
        o, L = _C.flash_attention_forward(q, k, v, scale, is_causal)

        # Save for backward
        ctx.save_for_backward(q, k, v, o, L)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass of Flash Attention.

        Uses the memory-efficient backward algorithm from metal-flash-attention.
        """
        try:
            from . import _C
        except ImportError:
            raise ImportError("MPS Flash Attention C extension not found.")

        q, k, v, o, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal

        # Call Metal backward kernels
        grad_q, grad_k, grad_v = _C.flash_attention_backward(
            grad_output.contiguous(),
            q, k, v, o, L,
            scale, is_causal
        )

        return grad_q, grad_k, grad_v, None, None


def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Compute scaled dot-product attention using Flash Attention on MPS.

    This function provides O(n) memory complexity instead of O(n²) by using
    the tiled Flash Attention algorithm implemented in Metal.

    Args:
        query: Query tensor of shape (B, num_heads, seq_len, head_dim)
        key: Key tensor of shape (B, num_heads, seq_len, head_dim)
        value: Value tensor of shape (B, num_heads, seq_len, head_dim)
        is_causal: If True, applies causal masking (for autoregressive models)
        scale: Scaling factor for attention scores. Default: 1/sqrt(head_dim)

    Returns:
        Output tensor of shape (B, num_heads, seq_len, head_dim)

    Example:
        >>> q = torch.randn(2, 8, 1024, 64, device='mps', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 1024, 64, device='mps', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 1024, 64, device='mps', dtype=torch.float16)
        >>> out = flash_attention(q, k, v)
        >>> out.shape
        torch.Size([2, 8, 1024, 64])
    """
    return FlashAttentionFunction.apply(query, key, value, scale, is_causal)


def flash_attention_backward(
    grad_output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    logsumexp: Tensor,
    scale: float,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute gradients for Flash Attention.

    This is the explicit backward pass, useful for custom autograd or debugging.
    Normally you don't need to call this directly - use flash_attention() which
    handles gradients automatically.

    Args:
        grad_output: Gradient w.r.t. output
        query, key, value: Input tensors from forward pass
        output: Output from forward pass
        logsumexp: Log-sum-exp values from forward pass
        scale: Attention scale used in forward
        is_causal: Whether causal masking was used

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    try:
        from . import _C
    except ImportError:
        raise ImportError("MPS Flash Attention C extension not found.")

    return _C.flash_attention_backward(
        grad_output.contiguous(),
        query, key, value, output, logsumexp,
        scale, is_causal
    )
