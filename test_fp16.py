#!/usr/bin/env python3
"""Test fp16 specifically"""

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

# Test with a fresh kernel (different size to avoid cache)
B, H, N, D = 1, 2, 128, 48  # Different dims

print(f"Testing fp16 with B={B}, H={H}, N={N}, D={D}")

q = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
k = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
v = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)

# Reference
scale = 1.0 / math.sqrt(D)
q_f = q.float()
k_f = k.float()
v_f = v.float()
attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
attn = F.softmax(attn, dim=-1)
ref = torch.matmul(attn, v_f)

# MFA
print("Running MFA...")
out = flash_attention(q, k, v)
torch.mps.synchronize()

print(f"Output has NaN: {torch.isnan(out).any().item()}")
print(f"Output mean: {out.mean().item()}")
print(f"Reference mean: {ref.mean().item()}")

if not torch.isnan(out).any():
    diff = (out.float() - ref).abs()
    print(f"Max diff: {diff.max().item()}")
    print(f"Mean diff: {diff.mean().item()}")
