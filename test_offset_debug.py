#!/usr/bin/env python3
"""Debug storage offsets"""

import os
import sys

os.environ["MFA_BRIDGE_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)
sys.path.insert(0, os.path.dirname(__file__))

import torch

# Test storage offsets
for dtype in [torch.float32, torch.float16]:
    print(f"\n{dtype}:")
    q = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)
    k = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)
    v = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)

    for name, t in [('Q', q), ('K', k), ('V', v)]:
        print(f"  {name}: shape={list(t.shape)}, storage_offset={t.storage_offset()}, "
              f"nbytes={t.nbytes}, element_size={t.element_size()}, "
              f"storage_nbytes={t.storage().nbytes()}")
