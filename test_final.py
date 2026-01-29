#!/usr/bin/env python3
"""
Final test: MPS Flash Attention with PyTorch
"""

import os
import sys

# Set the path to our MFA bridge dylib
os.environ["MFA_BRIDGE_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)

# Add module to path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import math
import time

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Import our module
try:
    from mps_flash_attn import flash_attention, is_available
    print(f"MPS Flash Attention available: {is_available()}")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_correctness(B, H, N, D, dtype=torch.float32):
    """Test that MFA produces correct results."""
    print(f"\nTest: B={B}, H={H}, N={N}, D={D}, dtype={dtype}")

    # Create inputs
    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # Reference: naive attention
    scale = 1.0 / math.sqrt(D)
    q_cpu = q.float().cpu()
    k_cpu = k.float().cpu()
    v_cpu = v.float().cpu()
    attn = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    ref = torch.matmul(attn, v_cpu)

    # MFA
    try:
        torch.mps.synchronize()
        out = flash_attention(q, k, v)
        torch.mps.synchronize()

        out_cpu = out.float().cpu()

        # Compare
        max_diff = (out_cpu - ref).abs().max().item()
        mean_diff = (out_cpu - ref).abs().mean().item()

        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        if max_diff < 0.01:  # Tolerance for fp16/fp32 differences
            print(f"  ✓ PASSED")
            return True
        else:
            print(f"  ✗ FAILED - difference too large")
            return False

    except Exception as e:
        print(f"  ✗ FAILED - {e}")
        return False

def benchmark(B, H, N, D, dtype=torch.float16, warmup=3, iters=10):
    """Benchmark MFA vs naive attention."""
    print(f"\nBenchmark: B={B}, H={H}, N={N}, D={D}")

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # Warmup
    for _ in range(warmup):
        try:
            _ = flash_attention(q, k, v)
            torch.mps.synchronize()
        except:
            pass

    # MFA timing
    try:
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            out = flash_attention(q, k, v)
            torch.mps.synchronize()
        mfa_time = (time.perf_counter() - start) / iters * 1000
        print(f"  MFA: {mfa_time:.2f} ms")
    except Exception as e:
        print(f"  MFA: FAILED - {e}")
        mfa_time = None

    # PyTorch SDPA timing
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
    sdpa_time = (time.perf_counter() - start) / iters * 1000
    print(f"  SDPA: {sdpa_time:.2f} ms")

    if mfa_time:
        speedup = sdpa_time / mfa_time
        print(f"  Speedup: {speedup:.2f}x")

# Run tests
print("\n" + "="*60)
print("CORRECTNESS TESTS")
print("="*60)

test_correctness(1, 1, 64, 32)
test_correctness(1, 4, 128, 64)
test_correctness(2, 8, 256, 64)

print("\n" + "="*60)
print("BENCHMARKS")
print("="*60)

benchmark(1, 8, 512, 64)
benchmark(1, 8, 1024, 64)
benchmark(1, 8, 2048, 64)
