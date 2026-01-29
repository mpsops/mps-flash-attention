# MPS Flash Attention - TODO

## Core Implementation

### Phase 1: Basic Forward Pass
- [ ] Create Swift wrapper around metal-flash-attention
  - [ ] Expose C-compatible API via `@_cdecl`
  - [ ] Handle AttentionDescriptor setup
  - [ ] Handle AttentionKernel creation and caching
  - [ ] Execute kernels on command buffer
- [ ] Bridge Swift → C++ via bridging header
- [ ] Implement `getMTLBuffer()` to extract Metal buffer from PyTorch tensor
  - [ ] Research PyTorch MPS internals (`storage().data()` approach)
  - [ ] Handle buffer offsets for batched attention
- [ ] Test forward pass correctness against naive attention

### Phase 2: Backward Pass
- [ ] Implement dQ kernel wrapper
- [ ] Implement dK/dV kernel wrapper
- [ ] Register with PyTorch autograd
- [ ] Test gradient correctness with `torch.autograd.gradcheck`

### Phase 3: Optimization
- [ ] Kernel caching (avoid JIT compile on every call)
- [ ] Batch processing (process all B*H heads in one dispatch)
- [ ] Support for different head dimensions (16, 32, 64, 128, 256)
- [ ] FP16/BF16 precision support
- [ ] Multi-head batching

### Phase 4: Integration
- [ ] Monkey-patch `F.scaled_dot_product_attention`
- [ ] Support attention masks
- [ ] Support dropout (if possible)
- [ ] Causal masking

## Build System

- [ ] Add metal-flash-attention as git submodule
- [ ] Setup Swift/C++ interop build
- [ ] Test on M1, M2, M3, M4 chips
- [ ] CI/CD for releases
- [ ] Wheel building for pip install

## Documentation

- [ ] API reference
- [ ] Performance benchmarks vs naive MPS attention
- [ ] Memory usage comparison
- [ ] Installation troubleshooting

## Research Questions

1. **Buffer access**: Can we directly use PyTorch MPS tensor's Metal buffer, or do we need to copy?
   - Looks like `storage().data()` gives us the buffer pointer
   - Need to verify with actual PyTorch MPS internals

2. **Synchronization**: How to properly synchronize between PyTorch MPS stream and our Metal commands?
   - Use `MPSStream` from ATen
   - Call `synchronize()` after our kernels

3. **Memory layout**: Does metal-flash-attention expect (B, H, N, D) or (B, N, H, D)?
   - MFA uses (N, D) for single-head
   - Need to handle multi-head batching ourselves

4. **Kernel caching**: metal-flash-attention JIT compiles kernels - how to cache effectively?
   - Create kernel once per (seq_len, head_dim, precision) combo
   - Use LRU cache in C++ layer

## Resources

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- [PyTorch MPS Extension Tutorial](https://medium.com/practical-coding/metal-shaders-with-pytorch-from-end-to-end-c95370b3449b)
- [TorchMPSCustomOpsDemo](https://github.com/grimoire/TorchMPSCustomOpsDemo)
- [PyTorch MPS Backend Docs](https://docs.pytorch.org/docs/stable/notes/mps.html)
