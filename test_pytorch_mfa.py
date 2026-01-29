#!/usr/bin/env python3
"""
Test MFA with PyTorch MPS tensors.

The key challenge is extracting the Metal buffer pointer from PyTorch MPS tensors.
PyTorch's MPS backend stores tensors in Metal buffers, but doesn't expose them directly.
"""

import ctypes
import os
import torch
import torch.nn.functional as F
import math

# Check MPS availability
if not torch.backends.mps.is_available():
    print("MPS not available!")
    exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Load the MFA dylib
dylib_path = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)
lib = ctypes.CDLL(dylib_path)

# Function signatures
lib.mfa_init.restype = ctypes.c_bool
lib.mfa_create_kernel.restype = ctypes.c_void_p
lib.mfa_create_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
lib.mfa_forward.restype = ctypes.c_bool
lib.mfa_forward.argtypes = [
    ctypes.c_void_p,  # kernel
    ctypes.c_void_p,  # q
    ctypes.c_void_p,  # k
    ctypes.c_void_p,  # v
    ctypes.c_void_p,  # o
    ctypes.c_void_p,  # l
    ctypes.c_int32,   # batch
    ctypes.c_int32,   # heads
]
lib.mfa_release_kernel.argtypes = [ctypes.c_void_p]

# Initialize MFA
print("\nInitializing MFA...")
assert lib.mfa_init(), "Failed to init MFA"

# Test parameters
B, H, N, D = 1, 1, 64, 32  # Start small
dtype = torch.float32

print(f"\nTest config: B={B}, H={H}, N={N}, D={D}, dtype={dtype}")

# Create test tensors
print("Creating MPS tensors...")
q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
v = torch.randn(B, H, N, D, device='mps', dtype=dtype)
o = torch.zeros(B, H, N, D, device='mps', dtype=dtype)
l = torch.zeros(B, H, N, device='mps', dtype=torch.float32)  # logsumexp

# Make contiguous
q = q.contiguous()
k = k.contiguous()
v = v.contiguous()

print(f"Q shape: {q.shape}, device: {q.device}")

# The tricky part: get Metal buffer pointers
# PyTorch doesn't expose this directly, but we can try via data_ptr()

def get_mps_ptr(tensor):
    """Get the data pointer of an MPS tensor."""
    # This returns the GPU memory address
    return tensor.data_ptr()

print("\nGetting buffer pointers...")
q_ptr = get_mps_ptr(q)
k_ptr = get_mps_ptr(k)
v_ptr = get_mps_ptr(v)
o_ptr = get_mps_ptr(o)
l_ptr = get_mps_ptr(l)

print(f"Q ptr: {hex(q_ptr)}")
print(f"K ptr: {hex(k_ptr)}")
print(f"V ptr: {hex(v_ptr)}")
print(f"O ptr: {hex(o_ptr)}")
print(f"L ptr: {hex(l_ptr)}")

# Note: data_ptr() gives us a raw GPU memory address, but MFA expects MTLBuffer objects.
# This won't work directly - we need to either:
# 1. Use PyTorch's internal MPS APIs to get the actual MTLBuffer
# 2. Create our own Metal buffers and copy data
# 3. Use a C++ extension that can access PyTorch internals

print("\n" + "="*60)
print("NOTE: Direct pointer passing won't work because:")
print("- data_ptr() returns raw GPU memory address")
print("- MFA expects MTLBuffer objects")
print("- Need C++ extension to bridge properly")
print("="*60)

# For now, let's just verify PyTorch's native attention works
print("\nRunning PyTorch native attention for comparison...")
torch.mps.synchronize()

# Reference attention
scale = 1.0 / math.sqrt(D)
attn = torch.matmul(q, k.transpose(-2, -1)) * scale
attn = F.softmax(attn, dim=-1)
out_ref = torch.matmul(attn, v)

torch.mps.synchronize()
print(f"Reference output shape: {out_ref.shape}")
print(f"Reference output mean: {out_ref.mean().item():.6f}")
print(f"Reference output std: {out_ref.std().item():.6f}")

# Try scaled_dot_product_attention
print("\nTrying F.scaled_dot_product_attention...")
try:
    out_sdpa = F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize()
    print(f"SDPA output shape: {out_sdpa.shape}")
    print(f"SDPA matches ref: {torch.allclose(out_sdpa, out_ref, rtol=1e-3, atol=1e-3)}")
except Exception as e:
    print(f"SDPA failed: {e}")

print("\n=== Next Steps ===")
print("To complete the integration, we need a C++ PyTorch extension that:")
print("1. Uses PyTorch's internal MPS APIs to get MTLBuffer from tensors")
print("2. Calls our MFA bridge with the actual Metal buffers")
print("3. Properly synchronizes with PyTorch's MPS stream")
