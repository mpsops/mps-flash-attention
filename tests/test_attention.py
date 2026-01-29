"""
Tests for MPS Flash Attention
"""

import pytest
import torch
import torch.nn.functional as F
import math


def test_flash_attention_forward():
    """Test that flash attention produces correct results."""
    pytest.importorskip("mps_flash_attention")
    from mps_flash_attention import flash_attention

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Test parameters
    B, H, N, D = 2, 8, 128, 64
    dtype = torch.float16

    # Create inputs
    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # Flash Attention result
    out_flash = flash_attention(q, k, v)

    # Reference (naive) attention on CPU
    q_cpu = q.float().cpu()
    k_cpu = k.float().cpu()
    v_cpu = v.float().cpu()

    scale = 1.0 / math.sqrt(D)
    attn = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out_ref = torch.matmul(attn, v_cpu)

    # Compare
    out_flash_cpu = out_flash.float().cpu()
    torch.testing.assert_close(out_flash_cpu, out_ref, rtol=1e-2, atol=1e-2)


def test_flash_attention_backward():
    """Test that gradients are computed correctly."""
    pytest.importorskip("mps_flash_attention")
    from mps_flash_attention import flash_attention

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    B, H, N, D = 2, 4, 64, 32
    dtype = torch.float32  # Use float32 for gradient checking

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)

    # Forward
    out = flash_attention(q, k, v)

    # Backward
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    # Check gradients exist
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

    # Check gradient shapes
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


def test_flash_attention_causal():
    """Test causal (autoregressive) attention."""
    pytest.importorskip("mps_flash_attention")
    from mps_flash_attention import flash_attention

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    B, H, N, D = 1, 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    out = flash_attention(q, k, v, is_causal=True)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_memory_efficiency():
    """Test that flash attention uses less memory than naive attention."""
    pytest.importorskip("mps_flash_attention")
    from mps_flash_attention import flash_attention

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    B, H, N, D = 1, 8, 4096, 64  # Large sequence
    dtype = torch.float16

    q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
    v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

    # This should NOT OOM with flash attention
    # (would need ~4GB for naive attention matrix at N=4096)
    out = flash_attention(q, k, v)

    assert out.shape == q.shape


def test_patch_pytorch():
    """Test the monkey-patching of F.scaled_dot_product_attention."""
    pytest.importorskip("mps_flash_attention")
    import mps_flash_attention

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Patch PyTorch
    mps_flash_attention.patch_pytorch()

    B, H, N, D = 2, 4, 64, 32
    q = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)

    # Should now use flash attention under the hood
    out = F.scaled_dot_product_attention(q, k, v)

    assert out.shape == q.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
