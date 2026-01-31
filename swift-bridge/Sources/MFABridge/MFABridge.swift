/**
 * MFA Bridge - C-callable API for metal-flash-attention
 *
 * This module provides PyTorch-friendly C functions to access Flash Attention on Metal.
 */

import Foundation
import Metal
import FlashAttention

// MARK: - Kernel Handle

/// Cached kernel with compiled pipelines for forward and backward passes
public final class MFAKernelCache {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let descriptor: AttentionDescriptor

    // Forward pass
    let forwardKernel: AttentionKernel
    let forwardPipeline: MTLComputePipelineState

    // Backward pass (optional)
    var backwardQueryKernel: AttentionKernel?
    var backwardQueryPipeline: MTLComputePipelineState?
    var backwardKVKernel: AttentionKernel?
    var backwardKVPipeline: MTLComputePipelineState?

    init(
        seqLenQ: Int,
        seqLenKV: Int,
        headDim: Int,
        lowPrecision: Bool,
        lowPrecisionOutputs: Bool,
        causal: Bool,
        hasMask: Bool,
        useBF16: Bool = false,
        windowSize: UInt32 = 0,
        quantizedKV: GEMMOperandPrecision? = nil,
        bf16Backward: Bool = false,
        device: MTLDevice
    ) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Configure descriptor
        var desc = AttentionDescriptor()
        desc.useBF16Inputs = useBF16
        // Mixed-precision backward requires lowPrecisionInputs=true for the backward kernel
        // to use the optimized FP16 parameter tables. When bf16Backward=true, we enable
        // low precision inputs for the backward kernels (they read FP32 but kernel expects FP16 config)
        desc.lowPrecisionInputs = (lowPrecision && !useBF16) || bf16Backward
        // Mixed-precision backward: when bf16Backward=true, use BF16 intermediates
        // This provides ~2x speedup on backward pass with minimal accuracy loss
        desc.lowPrecisionIntermediates = bf16Backward
        desc.lowPrecisionOutputs = lowPrecisionOutputs
        desc.useBF16Outputs = useBF16  // Match input precision for outputs
        desc.causal = causal
        desc.hasMask = hasMask
        // Sliding window attention: 0 = full attention, >0 = window size
        desc.windowSize = windowSize > 0 ? windowSize : nil
        // Quantized K/V: nil = standard precision
        desc.quantizedKV = quantizedKV
        desc.matrixDimensions = (
            row: UInt32(seqLenQ),
            column: UInt32(seqLenKV),
            head: UInt16(headDim)
        )
        desc.transposeState = (Q: false, K: false, V: false, O: false)
        self.descriptor = desc

        // Create forward kernel (using metallib + pipeline cache for instant loading)
        let fwdKernelDesc = desc.kernelDescriptor(type: .forward)
        self.forwardKernel = AttentionKernel(descriptor: fwdKernelDesc)
        let fwdSource = forwardKernel.createSource()

        let fwdConstants = MTLFunctionConstantValues()
        desc.setFunctionConstants(fwdConstants)

        // Use pipeline cache for zero-overhead loading
        let dims = desc.matrixDimensions!
        self.forwardPipeline = try MetallibCache.shared.getPipeline(
            source: fwdSource,
            functionName: "attention",
            constants: fwdConstants,
            rowDim: dims.row,
            colDim: dims.column,
            device: device
        )
    }

    func createBackwardKernels() throws {
        guard backwardQueryKernel == nil else { return }

        // Backward Query kernel
        let bwdQDesc = descriptor.kernelDescriptor(type: .backwardQuery)
        let bwdQKernel = AttentionKernel(descriptor: bwdQDesc)
        let bwdQSource = bwdQKernel.createSource()
        let bwdQLibrary = try device.makeLibraryWithFallback(source: bwdQSource)

        let bwdQConstants = MTLFunctionConstantValues()
        descriptor.setFunctionConstants(bwdQConstants)
        let bwdQFunction = try bwdQLibrary.makeFunction(name: "attention", constantValues: bwdQConstants)

        let bwdQPipelineDesc = MTLComputePipelineDescriptor()
        bwdQPipelineDesc.computeFunction = bwdQFunction
        bwdQPipelineDesc.maxTotalThreadsPerThreadgroup = 1024
        let bwdQPipeline = try device.makeComputePipelineState(
            descriptor: bwdQPipelineDesc, options: [], reflection: nil)

        self.backwardQueryKernel = bwdQKernel
        self.backwardQueryPipeline = bwdQPipeline

        // Backward KV kernel
        let bwdKVDesc = descriptor.kernelDescriptor(type: .backwardKeyValue)
        let bwdKVKernel = AttentionKernel(descriptor: bwdKVDesc)
        let bwdKVSource = bwdKVKernel.createSource()
        let bwdKVLibrary = try device.makeLibraryWithFallback(source: bwdKVSource)

        let bwdKVConstants = MTLFunctionConstantValues()
        descriptor.setFunctionConstants(bwdKVConstants)
        let bwdKVFunction = try bwdKVLibrary.makeFunction(name: "attention", constantValues: bwdKVConstants)

        let bwdKVPipelineDesc = MTLComputePipelineDescriptor()
        bwdKVPipelineDesc.computeFunction = bwdKVFunction
        bwdKVPipelineDesc.maxTotalThreadsPerThreadgroup = 1024
        let bwdKVPipeline = try device.makeComputePipelineState(
            descriptor: bwdKVPipelineDesc, options: [], reflection: nil)

        self.backwardKVKernel = bwdKVKernel
        self.backwardKVPipeline = bwdKVPipeline
    }
}

// MARK: - Global State

private var globalDevice: MTLDevice?
private var globalQueue: MTLCommandQueue?
private var kernelCache: [String: MFAKernelCache] = [:]
private let cacheLock = NSLock()

private func getDevice() -> MTLDevice {
    if globalDevice == nil {
        globalDevice = MTLCreateSystemDefaultDevice()!
        globalQueue = globalDevice!.makeCommandQueue()!
    }
    return globalDevice!
}

private func cacheKey(_ seqQ: Int, _ seqKV: Int, _ headDim: Int, _ lowPrec: Bool, _ lowPrecOut: Bool, _ causal: Bool, _ hasMask: Bool, _ useBF16: Bool, _ windowSize: UInt32 = 0, _ quantizedKV: UInt16 = 0) -> String {
    "\(seqQ)_\(seqKV)_\(headDim)_\(lowPrec)_\(lowPrecOut)_\(causal)_\(hasMask)_\(useBF16)_\(windowSize)_\(quantizedKV)"
}

// MARK: - C API

/// Initialize the MFA library. Call once at startup.
/// If preload=true, eagerly loads all cached pipelines into memory.
@_cdecl("mfa_init")
public func mfa_init() -> Bool {
    globalDevice = MTLCreateSystemDefaultDevice()
    guard globalDevice != nil else { return false }
    globalQueue = globalDevice!.makeCommandQueue()
    return globalQueue != nil
}

/// Preload cached pipelines into memory for zero-latency first call.
/// Call this at module import time (in a background thread ideally).
@_cdecl("mfa_preload_cache")
public func mfa_preload_cache() {
    guard let device = globalDevice ?? MTLCreateSystemDefaultDevice() else { return }
    if globalDevice == nil {
        globalDevice = device
        globalQueue = device.makeCommandQueue()
    }

    // Preload common configurations
    let seqLens = [512, 1024, 2048, 4096, 8192]
    let headDims = [64, 128]
    var configs: [(seqLen: Int, headDim: Int, lowPrecision: Bool)] = []

    for seqLen in seqLens {
        for headDim in headDims {
            configs.append((seqLen, headDim, true))  // fp16 most common
        }
    }

    MetallibCache.shared.preloadConfigurations(device: device, configurations: configs)
}

/// Create or get cached attention kernel.
/// Returns opaque handle, or nil on failure.
/// low_precision: if true, uses FP16 for inputs (ignored if use_bf16 is true)
/// low_precision_outputs: if true, output O is FP16/BF16 (saves memory)
/// causal: if true, applies causal masking (lower triangular attention)
/// has_mask: if true, expects a boolean attention mask buffer
/// use_bf16: if true, uses BF16 for inputs/outputs (avoids FP16 overflow)
@_cdecl("mfa_create_kernel")
public func mfa_create_kernel(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    low_precision: Bool,
    low_precision_outputs: Bool,
    causal: Bool,
    has_mask: Bool
) -> UnsafeMutableRawPointer? {
    // Legacy API - defaults to no BF16
    return mfa_create_kernel_v2(
        seq_len_q: seq_len_q,
        seq_len_kv: seq_len_kv,
        head_dim: head_dim,
        low_precision: low_precision,
        low_precision_outputs: low_precision_outputs,
        causal: causal,
        has_mask: has_mask,
        use_bf16: false
    )
}

/// Create or get cached attention kernel with BF16 support.
@_cdecl("mfa_create_kernel_v2")
public func mfa_create_kernel_v2(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    low_precision: Bool,
    low_precision_outputs: Bool,
    causal: Bool,
    has_mask: Bool,
    use_bf16: Bool
) -> UnsafeMutableRawPointer? {
    // Delegate to v3 with window_size=0 (full attention)
    return mfa_create_kernel_v3(
        seq_len_q: seq_len_q,
        seq_len_kv: seq_len_kv,
        head_dim: head_dim,
        low_precision: low_precision,
        low_precision_outputs: low_precision_outputs,
        causal: causal,
        has_mask: has_mask,
        use_bf16: use_bf16,
        window_size: 0
    )
}

/// Create or get cached attention kernel with sliding window support.
/// window_size: 0 = full attention, >0 = sliding window of that size
@_cdecl("mfa_create_kernel_v3")
public func mfa_create_kernel_v3(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    low_precision: Bool,
    low_precision_outputs: Bool,
    causal: Bool,
    has_mask: Bool,
    use_bf16: Bool,
    window_size: UInt32
) -> UnsafeMutableRawPointer? {
    let key = cacheKey(Int(seq_len_q), Int(seq_len_kv), Int(head_dim), low_precision, low_precision_outputs, causal, has_mask, use_bf16, window_size)

    cacheLock.lock()
    defer { cacheLock.unlock() }

    if let cached = kernelCache[key] {
        return Unmanaged.passRetained(cached).toOpaque()
    }

    do {
        let kernel = try MFAKernelCache(
            seqLenQ: Int(seq_len_q),
            seqLenKV: Int(seq_len_kv),
            headDim: Int(head_dim),
            lowPrecision: low_precision,
            lowPrecisionOutputs: low_precision_outputs,
            causal: causal,
            hasMask: has_mask,
            useBF16: use_bf16,
            windowSize: window_size,
            device: getDevice()
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
}

/// Create or get cached attention kernel with quantized K/V support.
/// quantized_kv: 0 = standard precision, 3 = FP8_E4M3, 4 = FP8_E5M2, 5 = INT8, 6 = NF4
@_cdecl("mfa_create_kernel_v4")
public func mfa_create_kernel_v4(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    low_precision: Bool,
    low_precision_outputs: Bool,
    causal: Bool,
    has_mask: Bool,
    use_bf16: Bool,
    window_size: UInt32,
    quantized_kv: UInt16
) -> UnsafeMutableRawPointer? {
    // Delegate to v5 with bf16_backward=false (default FP32 backward)
    return mfa_create_kernel_v5(
        seq_len_q: seq_len_q,
        seq_len_kv: seq_len_kv,
        head_dim: head_dim,
        low_precision: low_precision,
        low_precision_outputs: low_precision_outputs,
        causal: causal,
        has_mask: has_mask,
        use_bf16: use_bf16,
        window_size: window_size,
        quantized_kv: quantized_kv,
        bf16_backward: false
    )
}

/// Create or get cached attention kernel with all features including mixed-precision backward.
/// bf16_backward: if true, uses BF16 for backward pass intermediates (2x faster, slightly less accurate)
@_cdecl("mfa_create_kernel_v5")
public func mfa_create_kernel_v5(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    low_precision: Bool,
    low_precision_outputs: Bool,
    causal: Bool,
    has_mask: Bool,
    use_bf16: Bool,
    window_size: UInt32,
    quantized_kv: UInt16,
    bf16_backward: Bool
) -> UnsafeMutableRawPointer? {
    // Extended cache key including bf16_backward
    let key = "\(seq_len_q)_\(seq_len_kv)_\(head_dim)_\(low_precision)_\(low_precision_outputs)_\(causal)_\(has_mask)_\(use_bf16)_\(window_size)_\(quantized_kv)_\(bf16_backward)"

    cacheLock.lock()
    defer { cacheLock.unlock() }

    if let cached = kernelCache[key] {
        return Unmanaged.passRetained(cached).toOpaque()
    }

    // Convert quantized_kv to GEMMOperandPrecision
    var quantizedKVPrecision: GEMMOperandPrecision? = nil
    if quantized_kv > 0 {
        quantizedKVPrecision = GEMMOperandPrecision(rawValue: quantized_kv)
    }

    do {
        let kernel = try MFAKernelCache(
            seqLenQ: Int(seq_len_q),
            seqLenKV: Int(seq_len_kv),
            headDim: Int(head_dim),
            lowPrecision: low_precision,
            lowPrecisionOutputs: low_precision_outputs,
            causal: causal,
            hasMask: has_mask,
            useBF16: use_bf16,
            windowSize: window_size,
            quantizedKV: quantizedKVPrecision,
            bf16Backward: bf16_backward,
            device: getDevice()
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
}

/// Execute forward attention pass with quantized K/V using PyTorch's compute command encoder.
/// k_scale_ptr, v_scale_ptr: per-head scale factors for dequantization
@_cdecl("mfa_forward_encode_quantized")
public func mfa_forward_encode_quantized(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    k_scale_ptr: UnsafeMutableRawPointer,  // Per-head scales for K
    v_scale_ptr: UnsafeMutableRawPointer,  // Per-head scales for V
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64,
    k_offset: Int64,
    v_offset: Int64,
    o_offset: Int64,
    l_offset: Int64,
    k_scale_offset: Int64,
    v_scale_offset: Int64,
    mask_offset: Int64,
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()
    let encoder: MTLComputeCommandEncoder = unsafeBitCast(encoder_ptr, to: MTLComputeCommandEncoder.self)

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)
    let kScaleBuffer: MTLBuffer = unsafeBitCast(k_scale_ptr, to: MTLBuffer.self)
    let vScaleBuffer: MTLBuffer = unsafeBitCast(v_scale_ptr, to: MTLBuffer.self)

    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)

    // Q uses standard precision (FP16/BF16/FP32)
    let qElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    let outputElementSize = (cache.descriptor.lowPrecisionOutputs || cache.descriptor.useBF16Outputs) ? 2 : 4

    // K/V use quantized precision
    // For NF4: 2 values packed per byte along head dim, so kvHeadDim = headDim / 2
    // For FP8/INT8: 1 byte per element, so kvHeadDim = headDim
    let isNF4 = (cache.descriptor.quantizedKV == .NF4)
    let kvHeadDim = isNF4 ? (headDim / 2) : headDim
    let kvElementSize = 1  // All quantized types use 1 byte per packed unit

    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            let qHeadOffset = (b * Int(num_heads) + h) * seqLenQ * headDim * qElementSize
            let kvHeadOffset = (b * Int(num_heads) + h) * seqLenKV * kvHeadDim * kvElementSize
            let outputHeadOffset = (b * Int(num_heads) + h) * seqLenQ * headDim * outputElementSize
            let lHeadOffset = (b * Int(num_heads) + h) * seqLenQ * 4
            let scaleOffset = (b * Int(num_heads) + h) * 4  // One float per head
            let maskHeadOffset = (b * Int(num_heads) + h) * seqLenQ * seqLenKV

            encoder.setComputePipelineState(cache.forwardPipeline)
            encoder.setThreadgroupMemoryLength(
                Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

            encoder.setBuffer(qBuffer, offset: Int(q_offset) + qHeadOffset, index: 0)
            encoder.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder.setBuffer(oBuffer, offset: Int(o_offset) + outputHeadOffset, index: 3)
            encoder.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)

            if let maskBuffer = maskBuffer {
                encoder.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }

            // Set scale buffers for quantized K/V
            encoder.setBuffer(kScaleBuffer, offset: Int(k_scale_offset) + scaleOffset, index: 20)
            encoder.setBuffer(vScaleBuffer, offset: Int(v_scale_offset) + scaleOffset, index: 21)

            let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
                / Int(cache.forwardKernel.blockDimensions.parallelization)
            let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
            let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        }
    }

    return true
}

/// Execute forward attention pass using PyTorch's compute command encoder.
/// This is the zero-sync integration path - we encode directly onto PyTorch's encoder.
/// encoder_ptr: id<MTLComputeCommandEncoder> from PyTorch's stream->commandEncoder()
/// mask_ptr: optional attention mask buffer (nil if no mask)
@_cdecl("mfa_forward_encode")
public func mfa_forward_encode(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,  // id<MTLComputeCommandEncoder> from PyTorch
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,  // Optional mask buffer
    q_offset: Int64,
    k_offset: Int64,
    v_offset: Int64,
    o_offset: Int64,
    l_offset: Int64,
    mask_offset: Int64,
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()
    let encoder: MTLComputeCommandEncoder = unsafeBitCast(encoder_ptr, to: MTLComputeCommandEncoder.self)

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)

    // Mask buffer is optional
    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)
    // BF16 and FP16 both use 2 bytes, FP32 uses 4 bytes
    let inputElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    let outputElementSize = (cache.descriptor.lowPrecisionOutputs || cache.descriptor.useBF16Outputs) ? 2 : 4

    // Process each batch and head using the SAME encoder (kernel coalescing)
    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            // Q and O use seqLenQ, K and V use seqLenKV (they can differ!)
            let qHeadOffset = (b * Int(num_heads) + h) * seqLenQ * headDim * inputElementSize
            let kvHeadOffset = (b * Int(num_heads) + h) * seqLenKV * headDim * inputElementSize
            let outputHeadOffset = (b * Int(num_heads) + h) * seqLenQ * headDim * outputElementSize
            let lHeadOffset = (b * Int(num_heads) + h) * seqLenQ * 4
            // Mask is per-head: [B, H, seq_q, seq_k] stored as uchar
            let maskHeadOffset = (b * Int(num_heads) + h) * seqLenQ * seqLenKV

            encoder.setComputePipelineState(cache.forwardPipeline)
            encoder.setThreadgroupMemoryLength(
                Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

            encoder.setBuffer(qBuffer, offset: Int(q_offset) + qHeadOffset, index: 0)
            encoder.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder.setBuffer(oBuffer, offset: Int(o_offset) + outputHeadOffset, index: 3)
            encoder.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)

            // Set mask buffer if provided
            if let maskBuffer = maskBuffer {
                encoder.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }

            let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
                / Int(cache.forwardKernel.blockDimensions.parallelization)
            let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
            let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        }
    }

    // Don't end encoding - PyTorch manages the encoder lifecycle
    return true
}

/// Legacy forward function that creates its own command buffer (with sync).
/// Use mfa_forward_encode for zero-sync integration with PyTorch.
@_cdecl("mfa_forward")
public func mfa_forward(
    kernel_handle: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64,
    k_offset: Int64,
    v_offset: Int64,
    o_offset: Int64,
    l_offset: Int64,
    mask_offset: Int64,
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)

    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    guard let commandBuffer = cache.commandQueue.makeCommandBuffer() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)
    // BF16 and FP16 both use 2 bytes, FP32 uses 4 bytes
    let inputElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    let outputElementSize = (cache.descriptor.lowPrecisionOutputs || cache.descriptor.useBF16Outputs) ? 2 : 4

    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            // Q and O use seqLenQ, K and V use seqLenKV (they can differ!)
            let qHeadOffset = (b * Int(num_heads) + h) * seqLenQ * headDim * inputElementSize
            let kvHeadOffset = (b * Int(num_heads) + h) * seqLenKV * headDim * inputElementSize
            let outputHeadOffset = (b * Int(num_heads) + h) * seqLenQ * headDim * outputElementSize
            let lHeadOffset = (b * Int(num_heads) + h) * seqLenQ * 4
            let maskHeadOffset = (b * Int(num_heads) + h) * seqLenQ * seqLenKV

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return false
            }

            encoder.setComputePipelineState(cache.forwardPipeline)
            encoder.setThreadgroupMemoryLength(
                Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

            encoder.setBuffer(qBuffer, offset: Int(q_offset) + qHeadOffset, index: 0)
            encoder.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder.setBuffer(oBuffer, offset: Int(o_offset) + outputHeadOffset, index: 3)
            encoder.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)

            if let maskBuffer = maskBuffer {
                encoder.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }

            let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
                / Int(cache.forwardKernel.blockDimensions.parallelization)
            let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
            let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
            encoder.endEncoding()
        }
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.error == nil
}

/// Execute backward attention pass using PyTorch's compute command encoder.
/// This encodes both backwardQuery and backwardKeyValue kernels onto the encoder.
@_cdecl("mfa_backward_encode")
public func mfa_backward_encode(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,  // id<MTLComputeCommandEncoder> from PyTorch
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    do_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    d_ptr: UnsafeMutableRawPointer,
    dq_ptr: UnsafeMutableRawPointer,
    dk_ptr: UnsafeMutableRawPointer,
    dv_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64,
    k_offset: Int64,
    v_offset: Int64,
    o_offset: Int64,
    do_offset: Int64,
    l_offset: Int64,
    d_offset: Int64,
    dq_offset: Int64,
    dk_offset: Int64,
    dv_offset: Int64,
    mask_offset: Int64,
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()
    let encoder: MTLComputeCommandEncoder = unsafeBitCast(encoder_ptr, to: MTLComputeCommandEncoder.self)

    do {
        try cache.createBackwardKernels()
    } catch {
        print("MFA Error creating backward kernels: \(error)")
        return false
    }

    guard let bwdQKernel = cache.backwardQueryKernel,
          let bwdQPipeline = cache.backwardQueryPipeline,
          let bwdKVKernel = cache.backwardKVKernel,
          let bwdKVPipeline = cache.backwardKVPipeline else {
        return false
    }

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let doBuffer: MTLBuffer = unsafeBitCast(do_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)
    let dBuffer: MTLBuffer = unsafeBitCast(d_ptr, to: MTLBuffer.self)
    let dqBuffer: MTLBuffer = unsafeBitCast(dq_ptr, to: MTLBuffer.self)
    let dkBuffer: MTLBuffer = unsafeBitCast(dk_ptr, to: MTLBuffer.self)
    let dvBuffer: MTLBuffer = unsafeBitCast(dv_ptr, to: MTLBuffer.self)

    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)
    // BF16 and FP16 both use 2 bytes, FP32 uses 4 bytes
    let inputElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    let outputElementSize = (cache.descriptor.lowPrecisionOutputs || cache.descriptor.useBF16Outputs) ? 2 : 4
    // dO element size: when lowPrecisionInputs=true, dO is BF16 (2 bytes), otherwise FP32 (4 bytes)
    let dOElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    // L element size: when lowPrecisionIntermediates=true, L is FP16 (2 bytes), otherwise FP32 (4 bytes)
    let lElementSize = cache.descriptor.lowPrecisionIntermediates ? 2 : 4
    // D element size: when lowPrecisionIntermediates=true, D is BF16 (2 bytes), otherwise FP32 (4 bytes)
    let dElementSize = cache.descriptor.lowPrecisionIntermediates ? 2 : 4

    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            let bhIndex = b * Int(num_heads) + h
            let qkvHeadOffset = bhIndex * seqLenQ * headDim * inputElementSize
            let kvHeadOffset = bhIndex * seqLenKV * headDim * inputElementSize
            let oHeadOffset = bhIndex * seqLenQ * headDim * outputElementSize
            let doHeadOffset = bhIndex * seqLenQ * headDim * dOElementSize  // dO may be different size than O
            let lHeadOffset = bhIndex * seqLenQ * lElementSize
            let dHeadOffset = bhIndex * seqLenQ * dElementSize
            let dqHeadOffset = bhIndex * seqLenQ * headDim * 4
            let dkvHeadOffset = bhIndex * seqLenKV * headDim * 4
            let maskHeadOffset = bhIndex * seqLenQ * seqLenKV

            // Phase 1: Backward Query
            encoder.setComputePipelineState(bwdQPipeline)
            encoder.setThreadgroupMemoryLength(Int(bwdQKernel.threadgroupMemoryAllocation), index: 0)
            encoder.setBuffer(qBuffer, offset: Int(q_offset) + qkvHeadOffset, index: 0)
            encoder.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder.setBuffer(oBuffer, offset: Int(o_offset) + oHeadOffset, index: 3)
            encoder.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)
            encoder.setBuffer(dBuffer, offset: Int(d_offset) + dHeadOffset, index: 5)
            encoder.setBuffer(doBuffer, offset: Int(do_offset) + doHeadOffset, index: 6)
            encoder.setBuffer(dqBuffer, offset: Int(dq_offset) + dqHeadOffset, index: 9)
            if let maskBuffer = maskBuffer {
                encoder.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }

            let blockCount1 = (seqLenQ + Int(bwdQKernel.blockDimensions.parallelization) - 1)
                / Int(bwdQKernel.blockDimensions.parallelization)
            encoder.dispatchThreadgroups(
                MTLSize(width: blockCount1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Int(bwdQKernel.threadgroupSize), height: 1, depth: 1)
            )

            // Phase 2: Backward KV
            encoder.setComputePipelineState(bwdKVPipeline)
            encoder.setThreadgroupMemoryLength(Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)
            encoder.setBuffer(qBuffer, offset: Int(q_offset) + qkvHeadOffset, index: 0)
            encoder.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder.setBuffer(oBuffer, offset: Int(o_offset) + oHeadOffset, index: 3)
            encoder.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)
            encoder.setBuffer(dBuffer, offset: Int(d_offset) + dHeadOffset, index: 5)
            encoder.setBuffer(doBuffer, offset: Int(do_offset) + doHeadOffset, index: 6)
            encoder.setBuffer(dvBuffer, offset: Int(dv_offset) + dkvHeadOffset, index: 7)
            encoder.setBuffer(dkBuffer, offset: Int(dk_offset) + dkvHeadOffset, index: 8)
            if let maskBuffer = maskBuffer {
                encoder.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }

            let blockCount2 = (seqLenKV + Int(bwdKVKernel.blockDimensions.parallelization) - 1)
                / Int(bwdKVKernel.blockDimensions.parallelization)
            encoder.dispatchThreadgroups(
                MTLSize(width: blockCount2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Int(bwdKVKernel.threadgroupSize), height: 1, depth: 1)
            )
        }
    }

    return true
}

/// Legacy backward function that creates its own command buffer (with sync).
@_cdecl("mfa_backward")
public func mfa_backward(
    kernel_handle: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    do_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    d_ptr: UnsafeMutableRawPointer,
    dq_ptr: UnsafeMutableRawPointer,
    dk_ptr: UnsafeMutableRawPointer,
    dv_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64,
    k_offset: Int64,
    v_offset: Int64,
    o_offset: Int64,
    do_offset: Int64,
    l_offset: Int64,
    d_offset: Int64,
    dq_offset: Int64,
    dk_offset: Int64,
    dv_offset: Int64,
    mask_offset: Int64,
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()

    do {
        try cache.createBackwardKernels()
    } catch {
        print("MFA Error creating backward kernels: \(error)")
        return false
    }

    guard let bwdQKernel = cache.backwardQueryKernel,
          let bwdQPipeline = cache.backwardQueryPipeline,
          let bwdKVKernel = cache.backwardKVKernel,
          let bwdKVPipeline = cache.backwardKVPipeline else {
        return false
    }

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let doBuffer: MTLBuffer = unsafeBitCast(do_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)
    let dBuffer: MTLBuffer = unsafeBitCast(d_ptr, to: MTLBuffer.self)
    let dqBuffer: MTLBuffer = unsafeBitCast(dq_ptr, to: MTLBuffer.self)
    let dkBuffer: MTLBuffer = unsafeBitCast(dk_ptr, to: MTLBuffer.self)
    let dvBuffer: MTLBuffer = unsafeBitCast(dv_ptr, to: MTLBuffer.self)

    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    guard let commandBuffer = cache.commandQueue.makeCommandBuffer() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)
    // BF16 and FP16 both use 2 bytes, FP32 uses 4 bytes
    let inputElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    let outputElementSize = (cache.descriptor.lowPrecisionOutputs || cache.descriptor.useBF16Outputs) ? 2 : 4
    // dO element size: when lowPrecisionInputs=true, dO is BF16 (2 bytes), otherwise FP32 (4 bytes)
    let dOElementSize = (cache.descriptor.lowPrecisionInputs || cache.descriptor.useBF16Inputs) ? 2 : 4
    // L element size: when lowPrecisionIntermediates=true, L is FP16 (2 bytes), otherwise FP32 (4 bytes)
    let lElementSize = cache.descriptor.lowPrecisionIntermediates ? 2 : 4
    // D element size: when lowPrecisionIntermediates=true, D is BF16 (2 bytes), otherwise FP32 (4 bytes)
    let dElementSize = cache.descriptor.lowPrecisionIntermediates ? 2 : 4

    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            let bhIndex = b * Int(num_heads) + h
            let qkvHeadOffset = bhIndex * seqLenQ * headDim * inputElementSize
            let kvHeadOffset = bhIndex * seqLenKV * headDim * inputElementSize
            let oHeadOffset = bhIndex * seqLenQ * headDim * outputElementSize
            let doHeadOffset = bhIndex * seqLenQ * headDim * dOElementSize
            let lHeadOffset = bhIndex * seqLenQ * lElementSize
            let dHeadOffset = bhIndex * seqLenQ * dElementSize
            let dqHeadOffset = bhIndex * seqLenQ * headDim * 4
            let dkvHeadOffset = bhIndex * seqLenKV * headDim * 4
            let maskHeadOffset = bhIndex * seqLenQ * seqLenKV

            guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else { return false }
            encoder1.setComputePipelineState(bwdQPipeline)
            encoder1.setThreadgroupMemoryLength(Int(bwdQKernel.threadgroupMemoryAllocation), index: 0)
            encoder1.setBuffer(qBuffer, offset: Int(q_offset) + qkvHeadOffset, index: 0)
            encoder1.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder1.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder1.setBuffer(oBuffer, offset: Int(o_offset) + oHeadOffset, index: 3)
            encoder1.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)
            encoder1.setBuffer(dBuffer, offset: Int(d_offset) + dHeadOffset, index: 5)
            encoder1.setBuffer(doBuffer, offset: Int(do_offset) + doHeadOffset, index: 6)
            encoder1.setBuffer(dqBuffer, offset: Int(dq_offset) + dqHeadOffset, index: 9)
            if let maskBuffer = maskBuffer {
                encoder1.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }
            let blockCount1 = (seqLenQ + Int(bwdQKernel.blockDimensions.parallelization) - 1)
                / Int(bwdQKernel.blockDimensions.parallelization)
            encoder1.dispatchThreadgroups(
                MTLSize(width: blockCount1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Int(bwdQKernel.threadgroupSize), height: 1, depth: 1)
            )
            encoder1.endEncoding()

            guard let encoder2 = commandBuffer.makeComputeCommandEncoder() else { return false }
            encoder2.setComputePipelineState(bwdKVPipeline)
            encoder2.setThreadgroupMemoryLength(Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)
            encoder2.setBuffer(qBuffer, offset: Int(q_offset) + qkvHeadOffset, index: 0)
            encoder2.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder2.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder2.setBuffer(oBuffer, offset: Int(o_offset) + oHeadOffset, index: 3)
            encoder2.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)
            encoder2.setBuffer(dBuffer, offset: Int(d_offset) + dHeadOffset, index: 5)
            encoder2.setBuffer(doBuffer, offset: Int(do_offset) + doHeadOffset, index: 6)
            encoder2.setBuffer(dvBuffer, offset: Int(dv_offset) + dkvHeadOffset, index: 7)
            encoder2.setBuffer(dkBuffer, offset: Int(dk_offset) + dkvHeadOffset, index: 8)
            if let maskBuffer = maskBuffer {
                encoder2.setBuffer(maskBuffer, offset: Int(mask_offset) + maskHeadOffset, index: 10)
            }
            let blockCount2 = (seqLenKV + Int(bwdKVKernel.blockDimensions.parallelization) - 1)
                / Int(bwdKVKernel.blockDimensions.parallelization)
            encoder2.dispatchThreadgroups(
                MTLSize(width: blockCount2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Int(bwdKVKernel.threadgroupSize), height: 1, depth: 1)
            )
            encoder2.endEncoding()
        }
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.error == nil
}

/// Release kernel handle
@_cdecl("mfa_release_kernel")
public func mfa_release_kernel(kernel_handle: UnsafeMutableRawPointer) {
    Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).release()
}

/// Get library version
@_cdecl("mfa_version")
public func mfa_version() -> UnsafePointer<CChar>? {
    let version = "0.1.0"
    return (version as NSString).utf8String
}

/// Pre-compile kernels for common configurations to eliminate runtime compilation overhead.
/// Call this once at install time or first run.
@_cdecl("mfa_precompile")
public func mfa_precompile() {
    let device = getDevice()

    // Common configurations: (seqLen, headDim, lowPrecision)
    var configs: [(seqLen: Int, headDim: Int, lowPrecision: Bool)] = []

    // Common sequence lengths
    let seqLens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    // Common head dimensions
    let headDims = [32, 48, 64, 80, 96, 128]

    for seqLen in seqLens {
        for headDim in headDims {
            configs.append((seqLen, headDim, false))  // fp32
            configs.append((seqLen, headDim, true))   // fp16
        }
    }

    MetallibCache.shared.precompile(device: device, configurations: configs)
}

/// Clear the metallib cache
@_cdecl("mfa_clear_cache")
public func mfa_clear_cache() {
    MetallibCache.shared.clearCache()
}

/// Set directory containing pre-shipped kernel binaries.
/// Call this before any kernel operations for zero-compilation loading.
@_cdecl("mfa_set_kernels_dir")
public func mfa_set_kernels_dir(path: UnsafePointer<CChar>) {
    let pathString = String(cString: path)
    MetallibCache.shared.setShippedKernelsDir(pathString)
}
