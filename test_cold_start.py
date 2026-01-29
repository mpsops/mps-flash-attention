#!/usr/bin/env python3
"""Test cold start vs warm start performance"""

import os
import sys
import subprocess
import time

os.environ["MFA_BRIDGE_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "swift-bridge/.build/release/libMFABridge.dylib"
)

# Test script that runs in a fresh Python process
TEST_SCRIPT = '''
import os
import time
os.environ["MFA_BRIDGE_PATH"] = "{bridge_path}"
import sys
sys.path.insert(0, "{module_path}")

import torch
from mps_flash_attn import flash_attention

# Time first call (includes any remaining compilation)
q = torch.randn(1, 8, {seq_len}, 64, device='mps', dtype=torch.float16)
k = torch.randn(1, 8, {seq_len}, 64, device='mps', dtype=torch.float16)
v = torch.randn(1, 8, {seq_len}, 64, device='mps', dtype=torch.float16)

torch.mps.synchronize()
start = time.perf_counter()
out = flash_attention(q, k, v)
torch.mps.synchronize()
first_call = (time.perf_counter() - start) * 1000

# Time warm calls
times = []
for _ in range(5):
    torch.mps.synchronize()
    start = time.perf_counter()
    out = flash_attention(q, k, v)
    torch.mps.synchronize()
    times.append((time.perf_counter() - start) * 1000)

warm_avg = sum(times) / len(times)
print(f"{{first_call:.1f}},{{warm_avg:.2f}}")
'''

bridge_path = os.path.join(os.path.dirname(__file__), "swift-bridge/.build/release/libMFABridge.dylib")
module_path = os.path.dirname(__file__)

print("=" * 60)
print("Cold Start vs Warm Start Benchmark")
print("(Each test runs in a fresh Python process)")
print("=" * 60)

for seq_len in [1024, 2048, 4096]:
    script = TEST_SCRIPT.format(
        bridge_path=bridge_path,
        module_path=module_path,
        seq_len=seq_len
    )

    # Run in subprocess for true cold start
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "MFA_BRIDGE_PATH": bridge_path}
    )

    if result.returncode == 0:
        first, warm = result.stdout.strip().split(",")
        print(f"N={seq_len}: First call: {first} ms, Warm avg: {warm} ms")
    else:
        print(f"N={seq_len}: FAILED - {result.stderr[:200]}")
