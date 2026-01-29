# MPS Flash Attention

Flash Attention for PyTorch on Apple Silicon (M1/M2/M3/M4).

**O(N) memory** instead of O(N²), enabling 8K+ sequence lengths on unified memory.

## Features

- **Forward pass**: 2-5x faster than PyTorch SDPA
- **Backward pass**: Full gradient support for training (fp32 precision)
- **Causal masking**: Native kernel support (only 5% overhead)
- **Attention masks**: Full boolean mask support for arbitrary masking patterns
- **FP16/FP32**: Native fp16 output (no conversion overhead)
- **Pre-compiled kernels**: Zero-compilation cold start (~6ms)

## Performance

Tested on M1 Max, N=2048, B=4, H=8, D=64:

| Operation | MPS Flash Attn | PyTorch SDPA | Speedup |
|-----------|----------------|--------------|---------|
| Forward | 5.3ms | 15ms | 2.8x |
| Forward+Backward | 55ms | 108ms | 2.0x |
| Memory | 80MB | 592MB | 7.4x less |

## Installation

### Prerequisites

- macOS 14+ (Sonoma) or macOS 15+ (Sequoia)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.10+ with PyTorch 2.0+

### Build from source

```bash
# Clone with submodules
git clone --recursive https://github.com/mpsops/mps-flash-attention.git
cd mps-flash-attention

# Build Swift bridge
cd swift-bridge
swift build -c release
cd ..

# Install Python package
pip install -e .
```

### Set environment variable

```bash
export MFA_BRIDGE_PATH=/path/to/mps-flash-attention/swift-bridge/.build/release/libMFABridge.dylib
```

## Usage

### Basic usage

```python
from mps_flash_attn import flash_attention

# Standard attention (B, H, N, D)
q = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device='mps', dtype=torch.float16)

out = flash_attention(q, k, v)
```

### Causal masking (for autoregressive models)

```python
out = flash_attention(q, k, v, is_causal=True)
```

### Attention masks (for custom masking patterns)

```python
# Boolean mask: True = masked (don't attend), False = attend
mask = torch.zeros(B, 1, N, N, dtype=torch.bool, device='mps')
mask[:, :, :, 512:] = True  # Mask out positions after 512

out = flash_attention(q, k, v, attn_mask=mask)
```

### Training with gradients

```python
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

out = flash_attention(q, k, v, is_causal=True)
loss = out.sum()
loss.backward()  # Computes dQ, dK, dV
```

### Drop-in replacement for SDPA

```python
from mps_flash_attn import replace_sdpa

# Monkey-patch F.scaled_dot_product_attention
replace_sdpa()

# Now all attention ops use Flash Attention on MPS
```

## Architecture

```
+----------------------------------------------------------+
|                    Python API                            |
|              mps_flash_attn/__init__.py                  |
|         (flash_attention, autograd Function)             |
+----------------------------+-----------------------------+
                             |
+----------------------------v-----------------------------+
|                 C++ Extension                            |
|            mps_flash_attn/csrc/mps_flash_attn.mm         |
|    (PyTorch bindings, MTLBuffer handling, offsets)       |
+----------------------------+-----------------------------+
                             | dlopen + dlsym
+----------------------------v-----------------------------+
|                 Swift Bridge                             |
|         swift-bridge/Sources/MFABridge/                  |
|   (MFABridge.swift, MetallibCache.swift)                 |
|   @_cdecl exports: mfa_init, mfa_create_kernel,          |
|                    mfa_forward, mfa_backward             |
+----------------------------+-----------------------------+
                             |
+----------------------------v-----------------------------+
|              Metal Flash Attention                       |
|    metal-flash-attention/Sources/FlashAttention/         |
|     (AttentionDescriptor, AttentionKernel, etc.)         |
|                                                          |
|   Generates Metal shader source at runtime,              |
|   compiles to .metallib, caches pipelines                |
+----------------------------------------------------------+
```

## Project Structure

```
mps-flash-attention/
├── mps_flash_attn/              # Python package
│   ├── __init__.py              # Public API (flash_attention, replace_sdpa)
│   ├── csrc/
│   │   └── mps_flash_attn.mm    # PyTorch C++ extension
│   └── kernels/                 # Pre-compiled metallibs (optional)
│
├── swift-bridge/                # Swift -> C bridge
│   ├── Package.swift
│   └── Sources/MFABridge/
│       ├── MFABridge.swift      # C-callable API (@_cdecl)
│       └── MetallibCache.swift  # Disk caching for metallibs
│
├── metal-flash-attention/       # Upstream (git submodule)
│   └── Sources/FlashAttention/
│       └── Attention/
│           ├── AttentionDescriptor/  # Problem configuration
│           ├── AttentionKernel/      # Metal shader generation
│           └── ...
│
├── scripts/
│   └── build_metallibs.py       # Pre-compile kernels for distribution
│
└── setup.py                     # Python package setup
```

## Changes from upstream metal-flash-attention

We made the following modifications to `metal-flash-attention`:

### 1. macOS 15+ compatibility (MTLLibraryCompiler.swift)

Apple restricted `__asm` in runtime-compiled Metal shaders on macOS 15. We added a fallback that uses `xcrun metal` CLI compilation when runtime compilation fails.

### 2. Causal masking support

Added `causal` flag to AttentionDescriptor and kernel generation:

- `AttentionDescriptor.swift`: Added `causal: Bool` property
- `AttentionKernelDescriptor.swift`: Added `causal: Bool` property
- `AttentionKernel.swift`: Added `causal` field
- `AttentionKernel+Softmax.swift`: Added `maskCausal()` function
- `AttentionKernel+Source.swift`: Added causal masking to forward/backward loops

## Next Steps

### 1. PR to upstream metal-flash-attention

The macOS 15 fix and causal masking should be contributed back:

```bash
cd metal-flash-attention
git checkout -b macos15-causal-support
# Commit changes to:
#   - Sources/FlashAttention/Utilities/MTLLibraryCompiler.swift (new file)
#   - Sources/FlashAttention/Attention/AttentionDescriptor/*.swift
#   - Sources/FlashAttention/Attention/AttentionKernel/*.swift
git push origin macos15-causal-support
# Open PR at https://github.com/philipturner/metal-flash-attention
```

### 2. Publish mps-flash-attention to PyPI

```bash
# Add pyproject.toml with proper metadata
# Build wheel with pre-compiled Swift bridge
python -m build
twine upload dist/*
```

### 3. Pre-compile kernels for zero cold start

```bash
python scripts/build_metallibs.py
# Copies metallibs to mps_flash_attn/kernels/
# These get shipped with the wheel
```

## Current Status (Jan 2025)

**Working:**
- Forward pass (fp16/fp32)
- Backward pass (dQ, dK, dV gradients)
- Causal masking
- Metallib disk caching
- Pipeline binary caching (MTLBinaryArchive)

**Known limitations:**
- Sequence length must be divisible by block size (typically 64)
- Head dimension: Best with 32, 64, 96, 128
- No dropout

## Credits

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) by Philip Turner
- [Flash Attention](https://arxiv.org/abs/2205.14135) paper by Tri Dao et al.

## License

MIT
