#!/usr/bin/env python3
"""Debug: Check actual MTLBuffer details from C++ side"""

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

# Create separate tests to avoid any memory reuse issues
print("=" * 60)
print("Test 1: Create all tensors, then run MFA")
print("=" * 60)

from mps_flash_attn import flash_attention

# Force a clean state
torch.mps.empty_cache()
import gc
gc.collect()

# FP16 test
q = torch.randn(1, 1, 64, 32, device='mps', dtype=torch.float16)
k = torch.randn(1, 1, 64, 32, device='mps', dtype=torch.float16)
v = torch.randn(1, 1, 64, 32, device='mps', dtype=torch.float16)
torch.mps.synchronize()

print(f"Q data_ptr: {q.data_ptr()}")
print(f"K data_ptr: {k.data_ptr()}")
print(f"V data_ptr: {v.data_ptr()}")

# Check for overlapping memory
def ranges_overlap(ptr1, size1, ptr2, size2):
    return not (ptr1 + size1 <= ptr2 or ptr2 + size2 <= ptr1)

q_ptr, q_size = q.data_ptr(), q.nbytes
k_ptr, k_size = k.data_ptr(), k.nbytes
v_ptr, v_size = v.data_ptr(), v.nbytes

print(f"\nMemory ranges:")
print(f"  Q: {q_ptr:#x} - {q_ptr + q_size:#x}")
print(f"  K: {k_ptr:#x} - {k_ptr + k_size:#x}")
print(f"  V: {v_ptr:#x} - {v_ptr + v_size:#x}")

if ranges_overlap(q_ptr, q_size, k_ptr, k_size):
    print("WARNING: Q and K memory overlap!")
if ranges_overlap(q_ptr, q_size, v_ptr, v_size):
    print("WARNING: Q and V memory overlap!")
if ranges_overlap(k_ptr, k_size, v_ptr, v_size):
    print("WARNING: K and V memory overlap!")

out = flash_attention(q, k, v)
torch.mps.synchronize()

print(f"\nOutput NaN: {torch.isnan(out).any().item()}")
print(f"Output mean: {out.mean().item()}")

# Reference
scale = 1.0 / math.sqrt(32)
qf = q.float()
kf = k.float()
vf = v.float()
attn = F.softmax(torch.matmul(qf, kf.transpose(-2, -1)) * scale, dim=-1)
ref = torch.matmul(attn, vf)
print(f"Reference mean: {ref.mean().item()}")
