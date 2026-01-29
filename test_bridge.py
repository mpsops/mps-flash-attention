#!/usr/bin/env python3
"""
Quick test of the MFA bridge library via ctypes.
This verifies the C API works before we integrate with PyTorch.
"""

import ctypes
import os

# Load the dylib
dylib_path = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)

print(f"Loading: {dylib_path}")
lib = ctypes.CDLL(dylib_path)

# Define function signatures
lib.mfa_init.restype = ctypes.c_bool
lib.mfa_init.argtypes = []

lib.mfa_version.restype = ctypes.c_char_p
lib.mfa_version.argtypes = []

lib.mfa_create_kernel.restype = ctypes.c_void_p
lib.mfa_create_kernel.argtypes = [
    ctypes.c_int32,  # seq_len_q
    ctypes.c_int32,  # seq_len_kv
    ctypes.c_int32,  # head_dim
    ctypes.c_bool,   # low_precision
]

lib.mfa_release_kernel.restype = None
lib.mfa_release_kernel.argtypes = [ctypes.c_void_p]

# Test basic functions
print("\n=== MFA Bridge Test ===\n")

# Get version
version = lib.mfa_version()
print(f"Version: {version.decode() if version else 'N/A'}")

# Initialize
print("Initializing MFA...")
success = lib.mfa_init()
print(f"Init: {'SUCCESS' if success else 'FAILED'}")

if success:
    # Create a test kernel
    print("\nCreating attention kernel (seq=128, head=64, fp32)...")
    kernel = lib.mfa_create_kernel(128, 128, 64, False)
    if kernel:
        print(f"Kernel created: {hex(kernel)}")

        # Note: We can't test mfa_forward without actual Metal buffers
        # That requires PyTorch MPS integration

        print("Releasing kernel...")
        lib.mfa_release_kernel(kernel)
        print("Done!")
    else:
        print("FAILED to create kernel")

print("\n=== Test Complete ===")
