#!/usr/bin/env python3
"""
Test with large sequences - this is where Flash Attention shines!
"""

import os
import sys

os.environ["MFA_BRIDGE_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import time
import gc

print(f"PyTorch {torch.__version__}")

from mps_flash_attn import flash_attention

def test_large(N, H=8, D=64, dtype=torch.float16):
    B = 1
    print(f"\n{'='*60}")
    print(f"Sequence length: {N}, Heads: {H}, HeadDim: {D}")
    print(f"Attention matrix would be: {N}x{N} = {N*N:,} elements")
    print(f"Memory for attention matrix: {N*N * H * 2 / 1e9:.2f} GB (fp16)")
    print(f"{'='*60}")

    gc.collect()
    torch.mps.empty_cache()

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # Test MFA
    print("\nMPS Flash Attention:")
    try:
        torch.mps.synchronize()
        start = time.perf_counter()
        out_mfa = flash_attention(q, k, v)
        torch.mps.synchronize()
        mfa_time = (time.perf_counter() - start) * 1000
        print(f"  ✓ SUCCESS in {mfa_time:.1f} ms")
        print(f"  Output shape: {out_mfa.shape}")
        print(f"  Output mean: {out_mfa.mean().item():.4f}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # Test PyTorch SDPA
    print("\nPyTorch SDPA:")
    try:
        torch.mps.synchronize()
        start = time.perf_counter()
        out_sdpa = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
        sdpa_time = (time.perf_counter() - start) * 1000
        print(f"  ✓ SUCCESS in {sdpa_time:.1f} ms")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        sdpa_time = None

    if 'mfa_time' in dir() and sdpa_time:
        print(f"\nSpeedup: {sdpa_time/mfa_time:.2f}x")

# Test increasing sequence lengths
for N in [1024, 2048, 4096, 8192]:
    test_large(N)
