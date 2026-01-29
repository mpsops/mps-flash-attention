#!/usr/bin/env python3
"""
Find the memory limit for MPS attention.
"""

import torch
import torch.nn.functional as F
import gc

print(f"PyTorch version: {torch.__version__}")

def test_attention(N, H=8, D=64, dtype=torch.float16):
    """Test attention at given sequence length."""
    B = 1
    try:
        gc.collect()
        torch.mps.empty_cache()

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Try scaled_dot_product_attention
        out = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()

        # Calculate memory for attention matrix
        attn_matrix_bytes = B * H * N * N * (2 if dtype == torch.float16 else 4)
        attn_matrix_gb = attn_matrix_bytes / (1024**3)

        print(f"N={N:5d}: SUCCESS (attn matrix would be {attn_matrix_gb:.2f} GB)")
        del q, k, v, out
        return True
    except Exception as e:
        print(f"N={N:5d}: FAILED - {e}")
        return False

# Test increasing sequence lengths
print("\nTesting attention memory limits...")
print("H=8, D=64, dtype=float16")
print("-" * 60)

for N in [256, 512, 1024, 2048, 4096, 8192, 16384]:
    success = test_attention(N)
    if not success:
        break

print("\n" + "=" * 60)
print("Note: PyTorch MPS uses naive attention without Flash Attention")
print("Memory usage scales as O(N²) for the attention matrix")
print("=" * 60)
