# MPS Flash Attention

Flash Attention for PyTorch on Apple Silicon (M1/M2/M3/M4).

**O(N) memory** instead of O(N²), enabling 100K+ sequence lengths on unified memory.

## Performance

Benchmarked on Apple Silicon (M1/M2/M3/M4):

| Seq Length | vs PyTorch SDPA | Notes |
|------------|-----------------|-------|
| 1024 | 1.1-2.0x faster | Crossover point |
| 2048 | 1.7-3.7x faster | Sweet spot |
| 4096 | 2.0-3.9x faster | Peak performance |
| 8192+ | 3-4x faster | SDPA often OOMs |

Average speedup: **1.8x** across all configurations.

## Installation

```bash
pip install mps-flash-attn
```

### Build from source

```bash
git clone --recursive https://github.com/mpsops/mps-flash-attention.git
cd mps-flash-attention

# Build Swift bridge
cd swift-bridge && swift build -c release && cd ..

# Install
pip install -e .

# Set bridge path
export MFA_BRIDGE_PATH=$PWD/swift-bridge/.build/release/libMFABridge.dylib
```

## Usage

### Basic Attention

```python
from mps_flash_attn import flash_attention

# (B, H, N, D) format
q = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)

out = flash_attention(q, k, v)
```

### Causal Masking

```python
out = flash_attention(q, k, v, is_causal=True)
```

### Sliding Window (Mistral/Llama 3.2)

```python
# Only attend to last 4096 tokens
out = flash_attention(q, k, v, is_causal=True, window_size=4096)
```

### Quantized KV Cache (2-4x memory savings)

```python
from mps_flash_attn import flash_attention_fp8, quantize_kv_fp8

# Quantize K/V to FP8
k_quant, k_scale = quantize_kv_fp8(k)
v_quant, v_scale = quantize_kv_fp8(v)

# Run attention with quantized KV
out = flash_attention_fp8(q, k_quant, v_quant, k_scale, v_scale)
```

### 100K+ Long Sequences

```python
from mps_flash_attn import flash_attention_chunked

# Process 100K tokens without OOM
q = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
k = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)
v = torch.randn(1, 8, 100000, 64, device='mps', dtype=torch.float16)

out = flash_attention_chunked(q, k, v, chunk_size=8192)
```

### Drop-in SDPA Replacement

```python
from mps_flash_attn import replace_sdpa

replace_sdpa()  # Patches F.scaled_dot_product_attention

# Now all PyTorch attention uses Flash Attention on MPS
```

### torch.compile() Support

```python
from mps_flash_attn import register_custom_op

register_custom_op()

@torch.compile
def my_attention(q, k, v):
    return torch.ops.mfa.flash_attention(q, k, v, False, None, None)
```

### Training with BF16 Backward

```python
out = flash_attention(q, k, v, bf16_backward=True)  # 2x faster backward
loss = out.sum()
loss.backward()
```

### Fused QKV Projection + Attention

```python
from mps_flash_attn import flash_attention_qkv

x = torch.randn(2, 512, 256, device='mps', dtype=torch.float16)
w_q = torch.randn(256, 256, device='mps', dtype=torch.float16)
w_k = torch.randn(256, 256, device='mps', dtype=torch.float16)
w_v = torch.randn(256, 256, device='mps', dtype=torch.float16)

# Projects x through Q/K/V, runs attention, returns (B, N, D)
out = flash_attention_qkv(x, w_q, w_k, w_v, num_heads=4)

# With combined QKV weight (single GEMM, faster)
w_qkv = torch.randn(768, 256, device='mps', dtype=torch.float16)
out = flash_attention_qkv(x, w_q, w_k, w_v, w_qkv=w_qkv, num_heads=4)

# GQA: 8 query heads, 2 KV heads
w_q = torch.randn(512, 256, device='mps', dtype=torch.float16)
w_k = torch.randn(128, 256, device='mps', dtype=torch.float16)
w_v = torch.randn(128, 256, device='mps', dtype=torch.float16)
out = flash_attention_qkv(x, w_q, w_k, w_v, num_heads=8, num_kv_heads=2)
```

### LoRA Fusion

```python
from mps_flash_attn import flash_attention_lora

x = torch.randn(2, 512, 256, device='mps', dtype=torch.float16)
w_q = torch.randn(256, 256, device='mps', dtype=torch.float16)
w_k = torch.randn(256, 256, device='mps', dtype=torch.float16)
w_v = torch.randn(256, 256, device='mps', dtype=torch.float16)

# LoRA rank-16 adapters for Q and V (common fine-tuning setup)
lora_a_q = torch.randn(256, 16, device='mps', dtype=torch.float16)
lora_b_q = torch.randn(16, 256, device='mps', dtype=torch.float16)
lora_a_v = torch.randn(256, 16, device='mps', dtype=torch.float16)
lora_b_v = torch.randn(16, 256, device='mps', dtype=torch.float16)

out = flash_attention_lora(
    x, w_q, w_k, w_v,
    lora_a_q=lora_a_q, lora_b_q=lora_b_q,
    lora_a_v=lora_a_v, lora_b_v=lora_b_v,
    lora_scale=1.0, num_heads=4,
)
# Gradients flow to LoRA A/B matrices automatically
out.sum().backward()
```

### Pre-scaled Bias (SDPA Format)

```python
from mps_flash_attn import flash_attention_with_bias

# With sdpa_format=True, pass bias in PyTorch SDPA convention
# (no manual sqrt(D) scaling needed)
bias = torch.randn(1, 8, 64, 64, device='mps', dtype=torch.float16)
out = flash_attention_with_bias(q, k, v, bias, sdpa_format=True)
```

### Benchmarking

```bash
# Quick benchmark
python -m mps_flash_attn.benchmark --suite quick

# Full suite with report
python -m mps_flash_attn.benchmark --suite full --output report.html
```

```python
from mps_flash_attn.benchmark import run_suite, compare_vs_sdpa

results = run_suite(seq_lengths=[1024, 2048, 4096])
compare_vs_sdpa()
```

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| Forward pass | ✅ | FP16/BF16/FP32 |
| Backward pass | ✅ | Full gradient support |
| Causal masking | ✅ | Native kernel support |
| Attention masks | ✅ | Boolean masks |
| Sliding window | ✅ | For local attention models |
| GQA/MQA | ✅ | Grouped-query attention |
| Quantized KV | ✅ | FP8, INT8, NF4 |
| Chunked attention | ✅ | 100K+ tokens |
| torch.compile() | ✅ | Custom op backend |
| Dropout | ❌ | Not supported |

## Architecture

```
Python API (mps_flash_attn)
         │
    C++ Extension (mps_flash_attn.mm)
         │ dlopen
    Swift Bridge (MFABridge.swift)
         │
    Metal Flash Attention (kernel generation)
         │
    Metal GPU Shaders
```

## Requirements

- macOS 14+ (Sonoma) or macOS 15+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- PyTorch 2.0+

## TODO / Future Optimizations

- [x] **Batched kernel dispatch** - 3D grid dispatch with `MTLSize(blockCount, num_heads, batch_size)` handles all batch/heads in one kernel launch
- [x] **Fused QKV projection + attention** - `flash_attention_qkv()` fuses linear projections with attention, supports combined QKV weight and GQA
- [x] **Pre-scaled bias option** - `flash_attention_with_bias(..., sdpa_format=True)` auto-converts SDPA-convention bias
- [x] **LoRA fusion** - `flash_attention_lora()` fuses base projections + low-rank adapters + attention without materializing full-rank matrices

## Credits

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) by Philip Turner
- [Flash Attention](https://arxiv.org/abs/2205.14135) paper by Tri Dao et al.

## License

MIT
