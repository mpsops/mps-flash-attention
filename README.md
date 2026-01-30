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

## Credits

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) by Philip Turner
- [Flash Attention](https://arxiv.org/abs/2205.14135) paper by Tri Dao et al.

## License

MIT
