"""
MPS Flash Attention - PyTorch bindings for metal-flash-attention

Provides O(n) memory Flash Attention for Apple Silicon via Metal.
"""

__version__ = "0.1.0"

from .ops import flash_attention, flash_attention_backward

__all__ = [
    "flash_attention",
    "flash_attention_backward",
    "patch_pytorch",
]


def patch_pytorch():
    """
    Monkey-patch PyTorch's scaled_dot_product_attention to use Flash Attention on MPS.

    After calling this, F.scaled_dot_product_attention will automatically use
    the Metal Flash Attention kernel when running on MPS device.
    """
    import torch
    import torch.nn.functional as F

    _original_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # Use Flash Attention for MPS tensors
        if query.device.type == 'mps' and attn_mask is None and dropout_p == 0.0:
            return flash_attention(query, key, value, is_causal=is_causal, scale=scale)
        # Fall back to original for other cases
        return _original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)

    F.scaled_dot_product_attention = patched_sdpa
    print("Patched F.scaled_dot_product_attention to use MPS Flash Attention")
