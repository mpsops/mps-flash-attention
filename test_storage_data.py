#!/usr/bin/env python3
"""Debug storage data pointers"""

import torch

for dtype in [torch.float32, torch.float16]:
    print(f"\n{dtype}:")
    t = torch.randn(1, 1, 64, 32, device='mps', dtype=dtype)
    print(f"  data_ptr: {t.data_ptr():#x}")
    print(f"  storage data: {t.storage().data_ptr():#x}")
    print(f"  storage_offset: {t.storage_offset()}")
    print(f"  nbytes: {t.nbytes}")

    # Compute offset as PyTorch does
    offset = t.data_ptr() - t.storage().data_ptr()
    print(f"  computed offset: {offset}")
