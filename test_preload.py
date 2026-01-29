#!/usr/bin/env python3
"""Test cold start with background preloading"""

import os
import sys
import subprocess
import time

# Test script that imports module (triggers preload) then uses it
TEST_SCRIPT = '''
import os
import time
os.environ["MFA_BRIDGE_PATH"] = "{bridge_path}"
import sys
sys.path.insert(0, "{module_path}")

# Time the import (includes background preload start)
import_start = time.perf_counter()
from mps_flash_attn import flash_attention
import_time = (time.perf_counter() - import_start) * 1000

# Wait a bit for background preload to finish
time.sleep({wait_time})

import torch

# Time first call
q = torch.randn(1, 8, {seq_len}, 64, device='mps', dtype=torch.float16)
k = torch.randn(1, 8, {seq_len}, 64, device='mps', dtype=torch.float16)
v = torch.randn(1, 8, {seq_len}, 64, device='mps', dtype=torch.float16)

torch.mps.synchronize()
start = time.perf_counter()
out = flash_attention(q, k, v)
torch.mps.synchronize()
first_call = (time.perf_counter() - start) * 1000

# Time warm call
torch.mps.synchronize()
start = time.perf_counter()
out = flash_attention(q, k, v)
torch.mps.synchronize()
warm_call = (time.perf_counter() - start) * 1000

print(f"{{import_time:.1f}},{{first_call:.1f}},{{warm_call:.2f}}")
'''

bridge_path = os.path.join(os.path.dirname(__file__), "swift-bridge/.build/release/libMFABridge.dylib")
module_path = os.path.dirname(__file__)

print("=" * 70)
print("Cold Start with Background Preloading")
print("=" * 70)

for wait_time in [0, 0.1, 0.5]:
    print(f"\nWait {wait_time}s after import before first call:")
    for seq_len in [1024, 2048, 4096]:
        script = TEST_SCRIPT.format(
            bridge_path=bridge_path,
            module_path=module_path,
            seq_len=seq_len,
            wait_time=wait_time
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "MFA_BRIDGE_PATH": bridge_path}
        )

        if result.returncode == 0:
            import_t, first, warm = result.stdout.strip().split(",")
            print(f"  N={seq_len}: import={import_t}ms, first={first}ms, warm={warm}ms")
        else:
            print(f"  N={seq_len}: FAILED - {result.stderr[:100]}")
