"""
Comprehensive pytest test suite for mps-flash-attention v2.0

Tests all features:
- Basic forward/backward (FP32, FP16, BF16)
- Causal attention
- GQA (Grouped Query Attention)
- Sliding window attention
- Quantized attention (FP8, INT8)
- Chunked/streaming attention (100K+ tokens)
- BF16 mixed-precision backward
- torch.compile() custom op
"""

import pytest
import torch
import torch.nn.functional as F
import math

# Skip all tests if MPS not available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


@pytest.fixture(scope="module")
def mfa():
    """Import mps_flash_attn module."""
    import mps_flash_attn
    if not mps_flash_attn.is_available():
        pytest.skip("mps_flash_attn not available")
    return mps_flash_attn


def reference_attention(q, k, v, is_causal=False, attn_mask=None, window_size=0):
    """Reference implementation using PyTorch ops."""
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    seq_len_q = q.size(-2)
    seq_len_k = k.size(-2)

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))

    if window_size > 0:
        # Sliding window: mask positions where col < row - window_size
        row_idx = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seq_len_k, device=q.device).unsqueeze(0)
        window_mask = col_idx < (row_idx - window_size)
        attn = attn.masked_fill(window_mask, float('-inf'))

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask.bool(), float('-inf'))

    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)


# =============================================================================
# Basic Forward Tests
# =============================================================================

class TestForward:
    """Test basic forward pass functionality."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_forward_basic(self, mfa, dtype):
        """Test basic forward pass with different dtypes."""
        B, H, N, D = 2, 8, 128, 64

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention(q, k, v)

        assert output.shape == (B, H, N, D)
        assert output.dtype == dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_forward_correctness(self, mfa, dtype):
        """Test forward pass correctness against reference."""
        B, H, N, D = 1, 4, 64, 32

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output_mfa = mfa.flash_attention(q, k, v)
        output_ref = reference_attention(q, k, v)

        # Tolerance depends on dtype
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:  # BF16
            rtol, atol = 5e-2, 5e-2

        torch.testing.assert_close(output_mfa, output_ref, rtol=rtol, atol=atol)


# =============================================================================
# Backward Tests
# =============================================================================

class TestBackward:
    """Test backward pass functionality."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_backward_basic(self, mfa, dtype):
        """Test backward pass produces valid gradients."""
        B, H, N, D = 2, 4, 64, 32

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)

        output = mfa.flash_attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_backward_bf16_mixed_precision(self, mfa):
        """Test BF16 mixed-precision backward (faster backward)."""
        B, H, N, D = 2, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)

        # With bf16_backward=True for faster backward
        output = mfa.flash_attention(q, k, v, bf16_backward=True)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


# =============================================================================
# Causal Attention Tests
# =============================================================================

class TestCausal:
    """Test causal (autoregressive) attention."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_causal_basic(self, mfa, dtype):
        """Test causal attention produces valid output."""
        B, H, N, D = 2, 8, 128, 64

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention(q, k, v, is_causal=True)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_causal_correctness(self, mfa):
        """Test causal attention correctness."""
        B, H, N, D = 1, 4, 64, 32
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output_mfa = mfa.flash_attention(q, k, v, is_causal=True)
        output_ref = reference_attention(q, k, v, is_causal=True)

        torch.testing.assert_close(output_mfa, output_ref, rtol=1e-2, atol=1e-2)


# =============================================================================
# GQA (Grouped Query Attention) Tests
# =============================================================================

class TestGQA:
    """Test Grouped Query Attention (different Q/KV head counts)."""

    @pytest.mark.parametrize("h_q,h_kv", [(8, 2), (8, 4), (16, 4), (32, 8)])
    def test_gqa_various_ratios(self, mfa, h_q, h_kv):
        """Test GQA with various head ratios."""
        B, N, D = 2, 128, 64
        dtype = torch.float16

        q = torch.randn(B, h_q, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, h_kv, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, h_kv, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention(q, k, v)

        assert output.shape == (B, h_q, N, D)
        assert not torch.isnan(output).any()


# =============================================================================
# Sliding Window Attention Tests
# =============================================================================

class TestSlidingWindow:
    """Test sliding window attention (Mistral/Llama 3.2 style)."""

    @pytest.mark.parametrize("window_size", [256, 512, 1024])
    def test_sliding_window_basic(self, mfa, window_size):
        """Test sliding window attention produces valid output."""
        B, H, N, D = 1, 8, 2048, 64
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention(q, k, v, is_causal=True, window_size=window_size)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sliding_window_correctness(self, mfa):
        """Test sliding window correctness against reference implementation."""
        B, H, N, D = 1, 4, 128, 32
        window_size = 32
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output_mfa = mfa.flash_attention(q, k, v, is_causal=True, window_size=window_size)
        output_ref = reference_attention(q, k, v, is_causal=True, window_size=window_size)

        # Should match reference implementation
        torch.testing.assert_close(output_mfa, output_ref, rtol=5e-2, atol=5e-2)


# =============================================================================
# Attention Mask Tests
# =============================================================================

class TestAttnMask:
    """Test external attention mask support."""

    def test_mask_basic(self, mfa):
        """Test attention with external mask."""
        B, H, N, D = 2, 8, 128, 64
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Mask out second half of keys
        mask = torch.zeros(B, 1, N, N, dtype=torch.bool, device='mps')
        mask[:, :, :, N//2:] = True

        output = mfa.flash_attention(q, k, v, attn_mask=mask)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()


# =============================================================================
# Quantized Attention Tests
# =============================================================================

class TestQuantized:
    """Test quantized attention (FP8, INT8)."""

    def test_quantize_fp8(self, mfa):
        """Test FP8 quantization helper."""
        B, H, N, D = 1, 4, 128, 64
        dtype = torch.float16

        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        k_q, v_q, k_s, v_s = mfa.quantize_kv_fp8(k, v)

        assert k_q.dtype == torch.uint8
        assert v_q.dtype == torch.uint8
        assert k_s.dtype == torch.float32
        assert v_s.dtype == torch.float32
        assert k_q.shape == k.shape
        assert v_q.shape == v.shape

    def test_quantize_int8(self, mfa):
        """Test INT8 quantization helper."""
        B, H, N, D = 1, 4, 128, 64
        dtype = torch.float16

        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        k_q, v_q, k_s, v_s = mfa.quantize_kv_int8(k, v)

        assert k_q.dtype == torch.uint8
        assert v_q.dtype == torch.uint8
        assert k_s.dtype == torch.float32
        assert v_s.dtype == torch.float32

    def test_flash_attention_fp8(self, mfa):
        """Test FP8 quantized attention forward.

        FP8 quantized attention uses linear quantization (not actual FP8 bit format)
        with per-head scale factors. The kernel dequantizes K/V on-the-fly during
        attention computation.
        """
        B, H, N, D = 1, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        k_q, v_q, k_s, v_s = mfa.quantize_kv_fp8(k, v)
        output = mfa.flash_attention_fp8(q, k_q, v_q, k_s, v_s)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()

    def test_quantize_nf4(self, mfa):
        """Test NF4 quantization helper.

        NF4 packs 2 values per byte along head dimension, so output shape is (B,H,N,D//2).
        """
        B, H, N, D = 1, 4, 128, 64
        dtype = torch.float16

        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        k_q, v_q, k_s, v_s = mfa.quantize_kv_nf4(k, v)

        assert k_q.dtype == torch.uint8
        assert v_q.dtype == torch.uint8
        assert k_s.dtype == torch.float32
        assert v_s.dtype == torch.float32
        # NF4 packs 2 values per byte, so D dimension is halved
        assert k_q.shape == (B, H, N, D // 2)
        assert v_q.shape == (B, H, N, D // 2)

    def test_flash_attention_nf4(self, mfa):
        """Test NF4 quantized attention forward.

        NF4 uses a 16-value codebook for 4-bit quantization, packing 2 values per byte.
        This provides 4x memory reduction for K/V cache with acceptable accuracy loss.
        """
        B, H, N, D = 2, 8, 128, 64
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        k_q, v_q, k_s, v_s = mfa.quantize_kv_nf4(k, v)
        output = mfa.flash_attention_nf4(q, k_q, v_q, k_s, v_s)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_flash_attention_nf4_correctness(self, mfa):
        """Test NF4 attention correctness against reference.

        NF4 is 4-bit so we expect larger error than FP8/INT8, but output
        should still be in a reasonable range (max diff < 0.5).
        """
        B, H, N, D = 1, 4, 64, 64
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Reference
        ref = reference_attention(q, k, v)

        # NF4
        k_q, v_q, k_s, v_s = mfa.quantize_kv_nf4(k, v)
        output = mfa.flash_attention_nf4(q, k_q, v_q, k_s, v_s)

        max_diff = (ref - output).abs().max().item()
        mean_diff = (ref - output).abs().mean().item()

        # 4-bit quantization has larger error, but should be bounded
        assert max_diff < 0.5, f"NF4 max diff {max_diff} exceeds threshold 0.5"
        assert mean_diff < 0.1, f"NF4 mean diff {mean_diff} exceeds threshold 0.1"

    @pytest.mark.parametrize("config", [
        (1, 1, 32, 32),    # Small
        (1, 8, 128, 64),   # Medium
        (2, 16, 256, 64),  # Large
        (1, 8, 512, 128),  # Large head dim
    ])
    def test_flash_attention_nf4_various_sizes(self, mfa, config):
        """Test NF4 attention with various tensor sizes."""
        B, H, N, D = config
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        k_q, v_q, k_s, v_s = mfa.quantize_kv_nf4(k, v)
        output = mfa.flash_attention_nf4(q, k_q, v_q, k_s, v_s)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()


# =============================================================================
# Chunked/Streaming Attention Tests
# =============================================================================

class TestChunked:
    """Test chunked attention for long sequences (100K+ tokens)."""

    def test_chunked_basic(self, mfa):
        """Test chunked attention produces valid output."""
        B, H, N, D = 1, 4, 8192, 64
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention_chunked(q, k, v, chunk_size=2048)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any(), "Chunked attention produced NaN"
        assert not torch.isinf(output).any(), "Chunked attention produced Inf"

    def test_chunked_vs_regular(self, mfa):
        """Test chunked attention matches regular attention."""
        B, H, N, D = 1, 4, 1024, 64
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Small chunk to force multiple chunks
        output_chunked = mfa.flash_attention_chunked(q, k, v, chunk_size=256)
        output_regular = mfa.flash_attention(q, k, v)

        # Chunked has some numerical differences due to online softmax but should be close
        torch.testing.assert_close(output_chunked, output_regular, rtol=0.1, atol=0.1)

    @pytest.mark.slow
    def test_chunked_long_sequence(self, mfa):
        """Test chunked attention with very long sequence."""
        B, H, N, D = 1, 4, 32768, 64  # 32K tokens
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention_chunked(q, k, v, chunk_size=4096)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()


# =============================================================================
# Large Sequence Tests (Memory Efficiency)
# =============================================================================

class TestLargeSequence:
    """Test memory efficiency with large sequences."""

    @pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
    def test_large_sequence_forward(self, mfa, seq_len):
        """Test forward pass with large sequences."""
        B, H, D = 1, 8, 64
        dtype = torch.float16

        q = torch.randn(B, H, seq_len, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, seq_len, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, seq_len, D, device='mps', dtype=dtype)

        output = mfa.flash_attention(q, k, v)

        assert output.shape == (B, H, seq_len, D)
        assert not torch.isnan(output).any()

    @pytest.mark.slow
    def test_very_large_sequence(self, mfa):
        """Test with 8K sequence (should not OOM with flash attention)."""
        B, H, N, D = 1, 8, 8192, 64
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        output = mfa.flash_attention(q, k, v)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()


# =============================================================================
# Custom Op / torch.compile Tests
# =============================================================================

class TestCustomOp:
    """Test PyTorch custom op registration for torch.compile()."""

    def test_register_custom_op(self, mfa):
        """Test custom op registration."""
        result = mfa.register_custom_op()
        # May return False if torch.library not available
        assert result in [True, False]

    def test_torch_compile(self, mfa):
        """Test that flash_attention works with torch.compile.

        Note: Dynamo warns about not knowing how to trace the pybind11 function.
        This is expected - the function runs as a graph break (eager) within
        compiled code, which is the standard pattern for custom C++ extensions.
        """
        import warnings
        mfa.register_custom_op()

        B, H, N, D = 1, 4, 128, 64
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Reference output
        output_ref = mfa.flash_attention(q, k, v)

        @torch.compile
        def attention_fn(q, k, v):
            return torch.ops.mfa.flash_attention(q, k, v)

        # Suppress the expected Dynamo warning about pybind11 function
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dynamo does not know how to trace")
            output = attention_fn(q, k, v)

        assert output.shape == (B, H, N, D)
        # Verify outputs match (compiled should produce identical results)
        assert torch.allclose(output, output_ref, rtol=1e-3, atol=1e-3)


# =============================================================================
# SDPA Replacement Tests
# =============================================================================

class TestSDPAReplacement:
    """Test SDPA monkey-patching."""

    def test_replace_sdpa(self, mfa):
        """Test that replace_sdpa patches F.scaled_dot_product_attention."""
        mfa.replace_sdpa()

        B, H, N, D = 1, 8, 2048, 64  # >= 1536 to trigger MFA
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # This should now use MFA internally
        output = F.scaled_dot_product_attention(q, k, v)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()


# =============================================================================
# Benchmark Tests (sanity checks)
# =============================================================================

class TestBenchmark:
    """Sanity tests for benchmark module."""

    def test_benchmark_import(self, mfa):
        """Test benchmark module imports."""
        from mps_flash_attn import benchmark
        assert hasattr(benchmark, 'run_suite')
        assert hasattr(benchmark, 'compare_vs_sdpa')
        assert hasattr(benchmark, 'generate_report')

    def test_benchmark_config(self, mfa):
        """Test BenchmarkConfig creation."""
        from mps_flash_attn.benchmark import BenchmarkConfig
        config = BenchmarkConfig(
            batch_size=1,
            num_heads=8,
            seq_len=1024,
            head_dim=64,
            dtype="float16",
        )
        assert config.seq_len == 1024

    @pytest.mark.slow
    def test_benchmark_quick_run(self, mfa):
        """Test running a quick benchmark."""
        from mps_flash_attn.benchmark import run_suite
        results = run_suite(
            seq_lengths=[512],
            dtypes=["float16"],
            batch_sizes=[1],
            head_dims=[64],
            include_causal=False,
            include_window=False,
            verbose=False,
        )
        assert len(results) > 0
        assert results[0].mfa_time_ms is not None


# =============================================================================
# Edge Cases
# =============================================================================

# =============================================================================
# Attention Bias Tests
# =============================================================================

class TestAttentionBias:
    """Test flash_attention_with_bias functionality.

    Critical tests for bias handling - these ensure the FP16/FP32 type
    mismatch bug (fixed in v0.3.6) doesn't regress.
    """

    def reference_attention_with_bias(self, q, k, v, bias, scale=None):
        """Reference implementation with MFA-style bias (add before scale).

        MFA formula: softmax((Q @ K^T + bias) * scale) @ V
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1))
        attn = (attn + bias.float()) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_bias_forward_basic(self, mfa, dtype):
        """Test forward pass with bias produces valid output."""
        B, H, N, D = 2, 4, 64, 32

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        bias = torch.randn(B, H, N, N, device='mps', dtype=dtype)

        output = mfa.flash_attention_with_bias(q, k, v, bias)

        assert output.shape == (B, H, N, D)
        assert output.dtype == dtype
        assert not torch.isnan(output).any(), "Bias attention produced NaN"
        assert not torch.isinf(output).any(), "Bias attention produced Inf"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_bias_correctness(self, mfa, dtype):
        """Test bias attention correctness against reference implementation."""
        B, H, N, D = 1, 4, 32, 32

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        bias = torch.randn(B, H, N, N, device='mps', dtype=dtype) * 0.1  # Small bias

        output_mfa = mfa.flash_attention_with_bias(q, k, v, bias)
        output_ref = self.reference_attention_with_bias(q, k, v, bias)

        # Check outputs match
        max_diff = (output_mfa - output_ref).abs().max().item()

        if dtype == torch.float32:
            assert max_diff < 0.01, f"FP32 max diff {max_diff} exceeds threshold"
        else:
            assert max_diff < 0.05, f"FP16 max diff {max_diff} exceeds threshold"

    def test_bias_not_ignored(self, mfa):
        """Critical test: Ensure bias is NOT ignored (regression test for v0.3.6 bug)."""
        B, H, N, D = 1, 4, 64, 32
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Zero bias should match no-bias attention
        zero_bias = torch.zeros(B, H, N, N, device='mps', dtype=dtype)
        output_zero_bias = mfa.flash_attention_with_bias(q, k, v, zero_bias)
        output_no_bias = mfa.flash_attention(q, k, v)

        # Large bias should produce DIFFERENT output
        large_bias = torch.randn(B, H, N, N, device='mps', dtype=dtype) * 10.0
        output_large_bias = mfa.flash_attention_with_bias(q, k, v, large_bias)

        # Zero bias should approximately match no bias
        zero_diff = (output_zero_bias - output_no_bias).abs().max().item()
        assert zero_diff < 0.01, f"Zero bias should match no-bias, got diff {zero_diff}"

        # Large bias MUST produce different output
        large_diff = (output_large_bias - output_no_bias).abs().max().item()
        assert large_diff > 0.1, f"Bias appears to be ignored! Diff with large bias: {large_diff}"

    def test_bias_broadcast(self, mfa):
        """Test bias broadcasting (1, H, N, N) -> (B, H, N, N)."""
        B, H, N, D = 4, 8, 64, 32
        dtype = torch.float16

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype)

        # Bias with batch dim = 1 (should broadcast)
        bias = torch.randn(1, H, N, N, device='mps', dtype=dtype)

        output = mfa.flash_attention_with_bias(q, k, v, bias)

        assert output.shape == (B, H, N, D)
        assert not torch.isnan(output).any()

    def test_bias_repeat_count(self, mfa):
        """Test bias_repeat_count for window attention efficiency."""
        num_windows = 16
        H, ws, D = 4, 49, 32  # 7x7 window = 49 tokens
        dtype = torch.float16

        q = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype)
        k = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype)
        v = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype)

        # Single bias to repeat for all windows
        bias = torch.randn(1, H, ws, ws, device='mps', dtype=dtype)

        output = mfa.flash_attention_with_bias(q, k, v, bias, bias_repeat_count=num_windows)

        assert output.shape == (num_windows, H, ws, D)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_bias_backward(self, mfa, dtype):
        """Test backward pass with bias produces valid gradients."""
        B, H, N, D = 2, 4, 32, 32

        q = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        bias = torch.randn(B, H, N, N, device='mps', dtype=dtype)

        output = mfa.flash_attention_with_bias(q, k, v, bias)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None, "q.grad is None"
        assert k.grad is not None, "k.grad is None"
        assert v.grad is not None, "v.grad is None"
        assert not torch.isnan(q.grad).any(), f"q.grad has NaN"
        assert not torch.isnan(k.grad).any(), f"k.grad has NaN"
        assert not torch.isnan(v.grad).any(), f"v.grad has NaN"
        assert not torch.isinf(q.grad).any(), f"q.grad has Inf"
        assert not torch.isinf(k.grad).any(), f"k.grad has Inf"
        assert not torch.isinf(v.grad).any(), f"v.grad has Inf"

    def test_bias_backward_gradient_magnitude(self, mfa):
        """Test that bias backward gradients have reasonable magnitude."""
        B, H, N, D = 1, 4, 32, 32
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device='mps', dtype=dtype, requires_grad=True)
        bias = torch.randn(B, H, N, N, device='mps', dtype=dtype) * 0.1

        output = mfa.flash_attention_with_bias(q, k, v, bias)
        loss = output.sum()
        loss.backward()

        # Gradients should have reasonable magnitude (not exploding)
        q_grad_norm = q.grad.norm().item()
        k_grad_norm = k.grad.norm().item()
        v_grad_norm = v.grad.norm().item()

        # Sanity check: gradients shouldn't be too large or too small
        assert 0.01 < q_grad_norm < 1000, f"q.grad norm {q_grad_norm} seems wrong"
        assert 0.01 < k_grad_norm < 1000, f"k.grad norm {k_grad_norm} seems wrong"
        assert 0.01 < v_grad_norm < 1000, f"v.grad norm {v_grad_norm} seems wrong"

    def test_relative_position_bias_pattern(self, mfa):
        """Test bias pattern similar to Swin Transformer relative position bias."""
        B, H, ws, D = 1, 4, 49, 32  # 7x7 window
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(B, H, ws, D, device='mps', dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, ws, D, device='mps', dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, ws, D, device='mps', dtype=dtype, requires_grad=True)

        # Simulate relative position bias: (H, ws, ws)
        # In real use, this comes from a learnable table indexed by relative positions
        rel_pos_bias = torch.randn(H, ws, ws, device='mps', dtype=dtype) * 0.02
        rel_pos_bias = rel_pos_bias.unsqueeze(0)  # (1, H, ws, ws)

        # MFA formula: softmax((Q @ K^T + bias) * scale)
        # SDPA formula: softmax((Q @ K^T) * scale + bias)
        # Convert: bias_mfa = bias_sdpa * sqrt(D)
        scaled_bias = rel_pos_bias * math.sqrt(D)

        output = mfa.flash_attention_with_bias(q, k, v, scaled_bias)
        loss = output.sum()
        loss.backward()

        assert not torch.isnan(output).any(), "Output has NaN"
        assert not torch.isnan(q.grad).any(), "q.grad has NaN"
        assert not torch.isnan(k.grad).any(), "k.grad has NaN"
        assert not torch.isnan(v.grad).any(), "v.grad has NaN"

    def test_bias_dtype_conversion(self, mfa):
        """Test that bias is correctly converted to FP32 internally (v0.3.6 fix)."""
        B, H, N, D = 1, 4, 32, 32

        # Test with FP16 inputs (the common case that was broken)
        q = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='mps', dtype=torch.float16)
        bias = torch.randn(B, H, N, N, device='mps', dtype=torch.float16) * 0.5

        # This should work correctly (was broken before v0.3.6)
        output = mfa.flash_attention_with_bias(q, k, v, bias)

        # Verify output is valid
        assert not torch.isnan(output).any(), "FP16 bias attention produced NaN"
        assert not torch.isinf(output).any(), "FP16 bias attention produced Inf"

        # Verify bias actually affected the output
        output_no_bias = mfa.flash_attention(q, k, v)
        diff = (output - output_no_bias).abs().max().item()
        assert diff > 0.01, f"Bias seems to be ignored (diff={diff})"

    def test_bias_repeat_count_stress(self, mfa):
        """Stress test for bias_repeat_count - regression test for v0.3.7 buffer overflow fix.

        This test caught a critical bug where bias_repeat_count > 0 with broadcast bias (1, H, N, N)
        caused buffer overflow in Metal kernel. The kernel was computing offsets like:
            (batch % repeat_count) * num_heads * N * N + head * N * N
        But with biasBatchStride=0 (broadcast), this read past buffer bounds.

        The fix ensures that when biasBatchStride=0, only head stride is used:
            head * N * N
        """
        ws = 49  # 7x7 window size (Swin Transformer)
        num_windows = 16
        D = 32
        H = 4
        dtype = torch.float16

        nan_count = 0
        for i in range(20):
            torch.manual_seed(42 + i)

            q = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype, requires_grad=True)
            k = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype, requires_grad=True)
            v = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype, requires_grad=True)

            # Single bias pattern broadcast to all windows - this is the pattern that triggered the bug
            rel_pos_bias = torch.randn(H, ws, ws, device='mps', dtype=dtype) * 0.02
            scaled_bias = rel_pos_bias.unsqueeze(0) * math.sqrt(D)  # (1, H, ws, ws)

            out = mfa.flash_attention_with_bias(q, k, v, scaled_bias, bias_repeat_count=num_windows)
            loss = out.sum()
            loss.backward()

            has_nan = (
                torch.isnan(q.grad).any() or
                torch.isnan(k.grad).any() or
                torch.isnan(v.grad).any() or
                torch.isnan(out).any()
            )
            if has_nan:
                nan_count += 1

        assert nan_count == 0, f"bias_repeat_count stress test: {nan_count}/20 iterations had NaN"

    def test_bias_repeat_count_correctness(self, mfa):
        """Test that bias_repeat_count produces correct results matching expanded bias."""
        num_windows = 8
        H, ws, D = 4, 32, 32
        dtype = torch.float16

        torch.manual_seed(42)
        q = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype)
        k = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype)
        v = torch.randn(num_windows, H, ws, D, device='mps', dtype=dtype)

        # Single bias to repeat
        bias_single = torch.randn(1, H, ws, ws, device='mps', dtype=dtype) * 0.1

        # Method 1: Use bias_repeat_count
        out1 = mfa.flash_attention_with_bias(q, k, v, bias_single, bias_repeat_count=num_windows)

        # Method 2: Manually expand bias
        bias_expanded = bias_single.expand(num_windows, H, ws, ws).contiguous()
        out2 = mfa.flash_attention_with_bias(q, k, v, bias_expanded, bias_repeat_count=0)

        # Both should produce identical results
        max_diff = (out1 - out2).abs().max().item()
        assert max_diff < 0.01, f"bias_repeat_count result differs from expanded bias by {max_diff}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_batch_size_1(self, mfa):
        """Test with batch size 1."""
        output = mfa.flash_attention(
            torch.randn(1, 8, 128, 64, device='mps', dtype=torch.float16),
            torch.randn(1, 8, 128, 64, device='mps', dtype=torch.float16),
            torch.randn(1, 8, 128, 64, device='mps', dtype=torch.float16),
        )
        assert output.shape == (1, 8, 128, 64)

    def test_single_head(self, mfa):
        """Test with single head."""
        output = mfa.flash_attention(
            torch.randn(2, 1, 128, 64, device='mps', dtype=torch.float16),
            torch.randn(2, 1, 128, 64, device='mps', dtype=torch.float16),
            torch.randn(2, 1, 128, 64, device='mps', dtype=torch.float16),
        )
        assert output.shape == (2, 1, 128, 64)

    def test_small_head_dim(self, mfa):
        """Test with small head dimension."""
        output = mfa.flash_attention(
            torch.randn(2, 8, 128, 32, device='mps', dtype=torch.float16),
            torch.randn(2, 8, 128, 32, device='mps', dtype=torch.float16),
            torch.randn(2, 8, 128, 32, device='mps', dtype=torch.float16),
        )
        assert output.shape == (2, 8, 128, 32)

    def test_large_head_dim(self, mfa):
        """Test with large head dimension (128)."""
        output = mfa.flash_attention(
            torch.randn(1, 8, 128, 128, device='mps', dtype=torch.float16),
            torch.randn(1, 8, 128, 128, device='mps', dtype=torch.float16),
            torch.randn(1, 8, 128, 128, device='mps', dtype=torch.float16),
        )
        assert output.shape == (1, 8, 128, 128)

    def test_asymmetric_sequence_lengths(self, mfa):
        """Test with different Q and KV sequence lengths."""
        N_q, N_kv = 128, 256
        output = mfa.flash_attention(
            torch.randn(1, 8, N_q, 64, device='mps', dtype=torch.float16),
            torch.randn(1, 8, N_kv, 64, device='mps', dtype=torch.float16),
            torch.randn(1, 8, N_kv, 64, device='mps', dtype=torch.float16),
        )
        assert output.shape == (1, 8, N_q, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
