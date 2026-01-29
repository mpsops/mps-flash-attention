#!/usr/bin/env python3
"""Debug the nan issue"""

import os
import sys

os.environ["MFA_BRIDGE_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)
sys.path.insert(0, os.path.dirname(__file__))

import torch
from mps_flash_attn import flash_attention

# Test fp32 vs fp16
for dtype in [torch.float32, torch.float16]:
    print(f"\nTesting {dtype}:")
    q = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)
    k = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)
    v = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)

    out = flash_attention(q, k, v)
    torch.mps.synchronize()

    print(f"  Has NaN: {torch.isnan(out).any().item()}")
    print(f"  Mean: {out.mean().item()}")
    print(f"  Min: {out.min().item()}")
    print(f"  Max: {out.max().item()}")
