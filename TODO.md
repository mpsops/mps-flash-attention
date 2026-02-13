# MPS Flash Attention - TODO

## Core Implementation

### Phase 1: Basic Forward Pass ✅
- [x] Create Swift wrapper around metal-flash-attention
  - [x] Expose C-compatible API via `@_cdecl`
  - [x] Handle AttentionDescriptor setup
  - [x] Handle AttentionKernel creation and caching
  - [x] Execute kernels on command buffer
- [x] Bridge Swift → C++ via bridging header
- [x] Implement `getMTLBuffer()` to extract Metal buffer from PyTorch tensor
  - [x] Research PyTorch MPS internals (`storage().data()` approach)
  - [x] Handle buffer offsets for batched attention
- [x] Test forward pass correctness against naive attention

### Phase 2: Backward Pass ✅
- [x] Implement dQ kernel wrapper
- [x] Implement dK/dV kernel wrapper
- [x] Register with PyTorch autograd
- [x] Test gradient correctness with `torch.autograd.gradcheck`

### Phase 3: Optimization ✅
- [x] Kernel caching (avoid JIT compile on every call)
- [x] Batch processing (process all B*H heads in one dispatch) - 3D grid `MTLSize(blockCount, num_heads, batch_size)`
- [x] Support for different head dimensions (16, 32, 64, 128, 256)
- [x] FP16/BF16 precision support
- [x] Multi-head batching via BatchedParams struct
- [x] Fused QKV projection + attention (`flash_attention_qkv()`)
- [x] Pre-scaled bias option (`sdpa_format=True`)
- [x] LoRA fusion (`flash_attention_lora()`)

### Phase 4: Integration ✅
- [x] Monkey-patch `F.scaled_dot_product_attention`
- [x] Support attention masks
- [ ] Support dropout (if possible)
- [x] Causal masking

## Build System

- [x] Add metal-flash-attention as git submodule
- [x] Setup Swift/C++ interop build
- [ ] Test on M1, M2, M3, M4 chips
- [ ] CI/CD for releases
- [ ] Wheel building for pip install

## Documentation

- [x] API reference
- [x] Performance benchmarks vs naive MPS attention
- [ ] Memory usage comparison
- [ ] Installation troubleshooting

## Research Questions (Resolved)

1. **Buffer access**: Direct use of PyTorch MPS tensor's Metal buffer via `storage().data()` ✅
2. **Synchronization**: Uses PyTorch's shared MPS command encoder (zero-sync) ✅
3. **Memory layout**: MFA uses (N, D) per-head; multi-head handled via BatchedParams with 3D dispatch ✅
4. **Kernel caching**: Two-level cache (memory + disk) of compiled shaders via MetallibCache ✅

## Resources

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- [PyTorch MPS Extension Tutorial](https://medium.com/practical-coding/metal-shaders-with-pytorch-from-end-to-end-c95370b3449b)
- [TorchMPSCustomOpsDemo](https://github.com/grimoire/TorchMPSCustomOpsDemo)
- [PyTorch MPS Backend Docs](https://docs.pytorch.org/docs/stable/notes/mps.html)
