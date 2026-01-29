#!/usr/bin/env python3
"""Test if cloning fixes the overlap issue"""

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

torch.mps.empty_cache()

# FP16 test with cloning
q = torch.randn(1, 1, 64, 32, device='mps', dtype=torch.float16).clone()
k = torch.randn(1, 1, 64, 32, device='mps', dtype=torch.float16).clone()
v = torch.randn(1, 1, 64, 32, device='mps', dtype=torch.float16).clone()
torch.mps.synchronize()

print(f"Q data_ptr: {q.data_ptr():#x}, nbytes={q.nbytes}")
print(f"K data_ptr: {k.data_ptr():#x}, nbytes={k.nbytes}")
print(f"V data_ptr: {v.data_ptr():#x}, nbytes={v.nbytes}")

# Check for overlapping memory
def ranges_overlap(ptr1, size1, ptr2, size2):
    return not (ptr1 + size1 <= ptr2 or ptr2 + size2 <= ptr1)

if ranges_overlap(q.data_ptr(), q.nbytes, k.data_ptr(), k.nbytes):
    print("WARNING: Q and K memory overlap!")
if ranges_overlap(q.data_ptr(), q.nbytes, v.data_ptr(), v.nbytes):
    print("WARNING: Q and V memory overlap!")
if ranges_overlap(k.data_ptr(), k.nbytes, v.data_ptr(), v.nbytes):
    print("WARNING: K and V memory overlap!")

out = flash_attention(q, k, v)
torch.mps.synchronize()

print(f"\nOutput NaN: {torch.isnan(out).any().item()}")
print(f"Output mean: {out.mean().item()}")

# Reference
scale = 1.0 / math.sqrt(32)
attn = F.softmax(torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale, dim=-1)
ref = torch.matmul(attn, v.float())
print(f"Reference mean: {ref.mean().item()}")

if not torch.isnan(out).any():
    diff = (out.float() - ref).abs()
    print(f"Max diff: {diff.max().item()}")
