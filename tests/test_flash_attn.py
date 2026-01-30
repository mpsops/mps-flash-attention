"""Test mps-flash-attn: FP32, FP16, BF16 support with benchmarks"""

import torch
import torch.nn.functional as F
import math
import time

# Build and load the extension
print("Loading mps_flash_attn...")
from mps_flash_attn import flash_attention, is_available

print(f"MPS available: {is_available()}")

def reference_attention(q, k, v, is_causal=False, attn_mask=None):
    """Reference implementation using PyTorch ops."""
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if is_causal:
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask.bool(), float('-inf'))

    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)

def test_forward(dtype, name):
    """Test forward pass."""
    torch.manual_seed(42)

    B, H, N, D = 2, 8, 128, 64

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    output = flash_attention(q, k, v)

    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    ok = not has_nan and not has_inf

    shape_ok = output.shape == (B, H, N, D)

    print(f"  {name} forward: shape={output.shape}, dtype={output.dtype}, ok={ok and shape_ok}")
    return ok and shape_ok

def test_backward(dtype, name):
    """Test backward pass."""
    torch.manual_seed(42)

    B, H, N, D = 2, 4, 64, 32

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)

    output = flash_attention(q, k, v)
    loss = output.sum()
    loss.backward()

    grad_q_ok = q.grad is not None and not torch.isnan(q.grad).any()
    grad_k_ok = k.grad is not None and not torch.isnan(k.grad).any()
    grad_v_ok = v.grad is not None and not torch.isnan(v.grad).any()
    ok = grad_q_ok and grad_k_ok and grad_v_ok

    print(f"  {name} backward: q={grad_q_ok}, k={grad_k_ok}, v={grad_v_ok}")
    return ok

def test_causal(dtype, name):
    """Test causal attention."""
    torch.manual_seed(42)

    B, H, N, D = 2, 8, 128, 64

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    output = flash_attention(q, k, v, is_causal=True)

    ok = not torch.isnan(output).any() and not torch.isinf(output).any()
    print(f"  {name} causal: shape={output.shape}, ok={ok}")
    return ok

def test_gqa(dtype, name):
    """Test Grouped Query Attention (GQA)."""
    torch.manual_seed(42)

    B, H_q, H_kv, N, D = 2, 8, 2, 128, 64  # 8 query heads, 2 KV heads

    q = torch.randn(B, H_q, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H_kv, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H_kv, N, D, device='mps', dtype=dtype)

    output = flash_attention(q, k, v)

    ok = not torch.isnan(output).any() and not torch.isinf(output).any()
    shape_ok = output.shape == (B, H_q, N, D)
    print(f"  {name} GQA ({H_q}q/{H_kv}kv): shape={output.shape}, ok={ok and shape_ok}")
    return ok and shape_ok

def test_correctness(dtype, name):
    """Test correctness against reference implementation."""
    torch.manual_seed(42)

    B, H, N, D = 1, 4, 64, 32

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    output_flash = flash_attention(q, k, v)
    output_ref = reference_attention(q, k, v)

    diff = (output_flash.float() - output_ref.float()).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (output_ref.float().abs() + 1e-6)).mean().item() * 100

    # Tolerance depends on dtype
    if dtype == torch.float32:
        ok = rel_diff < 1.0
    elif dtype == torch.float16:
        ok = rel_diff < 2.0
    else:  # BF16
        ok = rel_diff < 5.0

    print(f"  {name} correctness: max_diff={max_diff:.6f}, rel_diff={rel_diff:.2f}%, ok={ok}")
    return ok

def compare_fp32_bf16():
    """Compare FP32 vs BF16 outputs."""
    torch.manual_seed(42)

    B, H, N, D = 1, 4, 64, 32

    q_fp32 = torch.randn(B, H, N, D, device='mps', dtype=torch.float32)
    k_fp32 = torch.randn(B, H, N, D, device='mps', dtype=torch.float32)
    v_fp32 = torch.randn(B, H, N, D, device='mps', dtype=torch.float32)

    output_fp32 = flash_attention(q_fp32, k_fp32, v_fp32)

    # BF16
    q_bf16 = q_fp32.to(torch.bfloat16)
    k_bf16 = k_fp32.to(torch.bfloat16)
    v_bf16 = v_fp32.to(torch.bfloat16)

    output_bf16 = flash_attention(q_bf16, k_bf16, v_bf16)

    diff = (output_fp32 - output_bf16.to(torch.float32)).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (output_fp32.abs() + 1e-6)).mean().item() * 100

    # Attention has softmax which amplifies small differences, 10% tolerance is acceptable
    ok = rel_diff < 10.0
    print(f"  FP32 vs BF16: max_diff={max_diff:.6f}, rel_diff={rel_diff:.2f}%, ok={ok}")
    return ok

def test_large_sequence():
    """Test with large sequence length (memory efficiency test)."""
    torch.manual_seed(42)

    B, H, N, D = 1, 8, 4096, 64  # Large sequence
    dtype = torch.float16

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # This should NOT OOM with flash attention
    output = flash_attention(q, k, v)

    ok = not torch.isnan(output).any() and not torch.isinf(output).any()
    shape_ok = output.shape == (B, H, N, D)

    print(f"  Large seq (N={N}): shape={output.shape}, ok={ok and shape_ok}")
    return ok and shape_ok

def benchmark(dtype, name, warmup=5, runs=20):
    """Benchmark forward pass."""
    torch.manual_seed(42)

    B, H, N, D = 4, 8, 512, 64

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    for _ in range(warmup):
        _ = flash_attention(q, k, v)
    torch.mps.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = flash_attention(q, k, v)
    torch.mps.synchronize()
    elapsed = time.time() - start

    ms = (elapsed / runs) * 1000
    print(f"  {name}: {ms:.2f} ms")
    return ms

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Testing mps-flash-attn")
    print("=" * 50)

    all_ok = True

    print("\n1. Forward pass:")
    all_ok &= test_forward(torch.float32, "FP32")
    all_ok &= test_forward(torch.float16, "FP16")
    all_ok &= test_forward(torch.bfloat16, "BF16")

    print("\n2. Backward pass:")
    all_ok &= test_backward(torch.float32, "FP32")
    all_ok &= test_backward(torch.float16, "FP16")
    all_ok &= test_backward(torch.bfloat16, "BF16")

    print("\n3. Causal attention:")
    all_ok &= test_causal(torch.float32, "FP32")
    all_ok &= test_causal(torch.float16, "FP16")
    all_ok &= test_causal(torch.bfloat16, "BF16")

    print("\n4. GQA (Grouped Query Attention):")
    all_ok &= test_gqa(torch.float32, "FP32")
    all_ok &= test_gqa(torch.float16, "FP16")
    all_ok &= test_gqa(torch.bfloat16, "BF16")

    print("\n5. Correctness vs reference:")
    all_ok &= test_correctness(torch.float32, "FP32")
    all_ok &= test_correctness(torch.float16, "FP16")
    all_ok &= test_correctness(torch.bfloat16, "BF16")

    print("\n6. FP32 vs BF16 comparison:")
    all_ok &= compare_fp32_bf16()

    print("\n7. Large sequence (memory efficiency):")
    all_ok &= test_large_sequence()

    print("\n8. Benchmarks (B=4, H=8, N=512, D=64):")
    benchmark(torch.float32, "FP32")
    benchmark(torch.float16, "FP16")
    benchmark(torch.bfloat16, "BF16")

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 50)
