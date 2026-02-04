"""
torch.compile support for mps-flash-attention custom ops.

This registers the low-level Metal ops with torch.library so that torch.compile
can trace through them instead of falling back to eager mode.

Usage:
    import mps_flash_attn.torch_ops  # Register ops on import

Then the ops are automatically used via torch.ops.mfa.*
"""

import torch
from torch.library import Library, impl

# Use register_fake (new name) or impl_abstract (old name) for compatibility
try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Create library namespace
mfa_lib = Library("mfa", "DEF")

# =============================================================================
# Op Definitions (matching the _C pybind11 signatures)
# =============================================================================

# forward: basic forward pass (no logsumexp)
mfa_lib.define(
    "forward(Tensor query, Tensor key, Tensor value, bool is_causal, Tensor? attn_mask, int window_size) -> Tensor"
)

# forward_with_lse: forward pass with logsumexp for backward
mfa_lib.define(
    "forward_with_lse(Tensor query, Tensor key, Tensor value, bool is_causal, Tensor? attn_mask, int window_size) -> (Tensor, Tensor)"
)

# forward_with_bias_lse: forward pass with bias and logsumexp
mfa_lib.define(
    "forward_with_bias_lse(Tensor query, Tensor key, Tensor value, Tensor attn_bias, bool is_causal, int window_size, int bias_repeat_count) -> (Tensor, Tensor)"
)

# backward: compute gradients for q, k, v
mfa_lib.define(
    "backward(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor output, Tensor logsumexp, bool is_causal, Tensor? attn_mask, int window_size, bool bf16_backward) -> (Tensor, Tensor, Tensor)"
)

# backward_with_bias: compute gradients with bias
mfa_lib.define(
    "backward_with_bias(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor output, Tensor logsumexp, Tensor attn_bias, bool is_causal, int window_size, int bias_repeat_count) -> (Tensor, Tensor, Tensor)"
)

# =============================================================================
# MPS Implementations (call the C++ kernels)
# =============================================================================

@impl(mfa_lib, "forward", "MPS")
def forward_mps(query, key, value, is_causal, attn_mask, window_size):
    from . import _C
    return _C.forward(query, key, value, is_causal, attn_mask, window_size)


@impl(mfa_lib, "forward_with_lse", "MPS")
def forward_with_lse_mps(query, key, value, is_causal, attn_mask, window_size):
    from . import _C
    return _C.forward_with_lse(query, key, value, is_causal, attn_mask, window_size)


@impl(mfa_lib, "forward_with_bias_lse", "MPS")
def forward_with_bias_lse_mps(query, key, value, attn_bias, is_causal, window_size, bias_repeat_count):
    from . import _C
    return _C.forward_with_bias_lse(query, key, value, attn_bias, is_causal, window_size, bias_repeat_count)


@impl(mfa_lib, "backward", "MPS")
def backward_mps(grad_output, query, key, value, output, logsumexp, is_causal, attn_mask, window_size, bf16_backward):
    from . import _C
    return _C.backward(grad_output, query, key, value, output, logsumexp, is_causal, attn_mask, window_size, bf16_backward)


@impl(mfa_lib, "backward_with_bias", "MPS")
def backward_with_bias_mps(grad_output, query, key, value, output, logsumexp, attn_bias, is_causal, window_size, bias_repeat_count):
    from . import _C
    return _C.backward_with_bias(grad_output, query, key, value, output, logsumexp, attn_bias, is_causal, window_size, bias_repeat_count)


# =============================================================================
# Meta/Fake Implementations (for tracing - shapes only, no compute)
# =============================================================================

@register_fake("mfa::forward")
def forward_meta(query, key, value, is_causal, attn_mask, window_size):
    # query: (B, H, N, D) -> output: (B, H, N, D)
    return query.new_empty(query.shape)


@register_fake("mfa::forward_with_lse")
def forward_with_lse_meta(query, key, value, is_causal, attn_mask, window_size):
    # query: (B, H, N, D) -> output: (B, H, N, D), logsumexp: (B, H, N)
    B, H, N, D = query.shape
    output = query.new_empty(query.shape)
    logsumexp = query.new_empty(B, H, N)
    return output, logsumexp


@register_fake("mfa::forward_with_bias_lse")
def forward_with_bias_lse_meta(query, key, value, attn_bias, is_causal, window_size, bias_repeat_count):
    # query: (B, H, N, D) -> output: (B, H, N, D), logsumexp: (B, H, N)
    B, H, N, D = query.shape
    output = query.new_empty(query.shape)
    logsumexp = query.new_empty(B, H, N)
    return output, logsumexp


@register_fake("mfa::backward")
def backward_meta(grad_output, query, key, value, output, logsumexp, is_causal, attn_mask, window_size, bf16_backward):
    # Returns dQ, dK, dV with same shapes as Q, K, V
    dQ = query.new_empty(query.shape)
    dK = key.new_empty(key.shape)
    dV = value.new_empty(value.shape)
    return dQ, dK, dV


@register_fake("mfa::backward_with_bias")
def backward_with_bias_meta(grad_output, query, key, value, output, logsumexp, attn_bias, is_causal, window_size, bias_repeat_count):
    # Returns dQ, dK, dV with same shapes as Q, K, V
    dQ = query.new_empty(query.shape)
    dK = key.new_empty(key.shape)
    dV = value.new_empty(value.shape)
    return dQ, dK, dV


print("✓ mps_flash_attn torch.compile ops registered")
