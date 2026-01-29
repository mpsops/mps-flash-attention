#!/usr/bin/env python3
"""Test accuracy for both fp32 and fp16"""

import os
import sys

os.environ["MFA_BRIDGE_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import math

from mps_flash_attn import flash_attention

def test_accuracy(B, H, N, D, dtype):
    print(f"\n{'='*60}")
    print(f"Testing B={B}, H={H}, N={N}, D={D}, dtype={dtype}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # Reference (in fp32 for accuracy)
    scale = 1.0 / math.sqrt(D)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    attn = F.softmax(torch.matmul(qf, kf.transpose(-2, -1)) * scale, dim=-1)
    ref = torch.matmul(attn, vf)

    # MFA
    out = flash_attention(q, k, v)
    torch.mps.synchronize()

    # Compare
    out_f = out.float()
    diff = (out_f - ref).abs()

    print(f"  Output dtype: {out.dtype}")
    print(f"  Output mean: {out.mean().item():.6f}")
    print(f"  Reference mean: {ref.mean().item():.6f}")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")

    # Expected tolerance depends on precision
    if dtype == torch.float32:
        tolerance = 1e-4
    else:
        tolerance = 5e-2  # fp16 has lower precision

    if diff.max().item() < tolerance:
        print(f"  ✓ PASSED (within {tolerance} tolerance)")
        return True
    else:
        print(f"  ✗ FAILED (max diff {diff.max().item()} > {tolerance})")
        return False

# Run tests
all_passed = True
for dtype in [torch.float32, torch.float16]:
    for B, H, N, D in [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (1, 8, 256, 128),
        (1, 2, 512, 48),
    ]:
        if not test_accuracy(B, H, N, D, dtype):
            all_passed = False

print(f"\n{'='*60}")
if all_passed:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED!")
print(f"{'='*60}")
