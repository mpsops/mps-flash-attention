#!/usr/bin/env python3
"""Benchmark with warm-up (no compilation overhead)"""

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

from mps_flash_attn import flash_attention

def benchmark(N, H=8, D=64, dtype=torch.float16, warmup=3, runs=10):
    B = 1
    print(f"\nN={N}, H={H}, D={D}, dtype={dtype}")

    gc.collect()
    torch.mps.empty_cache()

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # Warmup MFA
    for _ in range(warmup):
        out = flash_attention(q, k, v)
        torch.mps.synchronize()

    # Benchmark MFA
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        out = flash_attention(q, k, v)
        torch.mps.synchronize()
    mfa_time = (time.perf_counter() - start) / runs * 1000

    # Warmup SDPA
    for _ in range(warmup):
        out = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()

    # Benchmark SDPA
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        out = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
    sdpa_time = (time.perf_counter() - start) / runs * 1000

    speedup = sdpa_time / mfa_time
    print(f"  MFA:  {mfa_time:.2f} ms")
    print(f"  SDPA: {sdpa_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x {'✓' if speedup > 1 else ''}")
    return mfa_time, sdpa_time, speedup

print("=" * 60)
print("Flash Attention Benchmark (warm start, 10 runs average)")
print("=" * 60)

for N in [512, 1024, 2048, 4096, 8192]:
    benchmark(N)
