/**
 * MFA Bridge - C-callable API for metal-flash-attention
 *
 * This module provides PyTorch-friendly C functions to access Flash Attention on Metal.
 *
 * BATCHED DISPATCH: All encode functions use 3D dispatch (blocks x heads x batch)
 * instead of nested loops. This dramatically improves performance for high-batch
 * workloads (e.g., 484 batches goes from 2904 dispatches to 1 dispatch).
 *
 * MONOLITHIC IR: Kernels are compiled via MetalASM (in-process LLVM IR → metallib).
 * No xcrun, no Metal CLI tools, no disk caching. All dimensions baked as literals.
 */

import Foundation
import Metal
import FlashAttention

// MARK: - Kernel Handle

/// Cached kernel with compiled pipelines for forward and backward passes
public final class MFAKernelCache {
    let descriptor: AttentionDescriptor

    // Forward pass
    let forwardKernel: AttentionKernel
    let forwardPipeline: MTLComputePipelineState

    // Backward pass (lazy)
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
        hasAttnBias: Bool = false,
        biasBatchStride: UInt32 = 0,
        biasHeadStride: UInt32 = 0,
        biasRepeatCount: UInt32 = 0
    ) throws {
        // Configure descriptor
        var desc = AttentionDescriptor()
        desc.useBF16Inputs = useBF16
        desc.lowPrecisionInputs = (lowPrecision && !useBF16) || bf16Backward
        desc.lowPrecisionIntermediates = bf16Backward
        desc.lowPrecisionOutputs = lowPrecisionOutputs
        desc.useBF16Outputs = useBF16
        desc.causal = causal
        desc.hasMask = hasMask
        desc.hasAttnBias = hasAttnBias
        desc.biasBatchStride = biasBatchStride
        desc.biasHeadStride = biasHeadStride
        desc.biasRepeatCount = biasRepeatCount
        desc.windowSize = windowSize > 0 ? windowSize : nil
        desc.quantizedKV = quantizedKV
        desc.matrixDimensions = (
            row: UInt32(seqLenQ),
            column: UInt32(seqLenKV),
            head: UInt16(headDim)
        )
        desc.transposeState = (Q: false, K: false, V: false, O: false)
        self.descriptor = desc

        // Create forward kernel via monolithic IR pipeline
        let fwd = AttentionKernel.pipeline(for: desc, type: .forward)
        self.forwardKernel = fwd.kernel
        self.forwardPipeline = fwd.pipeline
    }

    func createBackwardKernels() throws {
        guard backwardQueryKernel == nil else { return }

        let bwdQ = AttentionKernel.pipeline(for: descriptor, type: .backwardQuery)
        self.backwardQueryKernel = bwdQ.kernel
        self.backwardQueryPipeline = bwdQ.pipeline

        let bwdKV = AttentionKernel.pipeline(for: descriptor, type: .backwardKeyValue)
        self.backwardKVKernel = bwdKV.kernel
        self.backwardKVPipeline = bwdKV.pipeline
    }

    /// Create batched params buffer using the monolithic IR layout (15 x UInt32)
    func createBatchedParamsBuffer(numHeads: Int) -> MTLBuffer {
        let dims = descriptor.matrixDimensions!
        return AttentionKernel.createBatchedParamsBuffer(
            numHeads: UInt32(numHeads),
            R: dims.row, C: dims.column, D: UInt32(dims.head),
            quantizedKV: descriptor.quantizedKV
        )
    }
}

// MARK: - Global State

private var kernelCache: [String: MFAKernelCache] = [:]
private let cacheLock = NSLock()

// MARK: - C API

/// Initialize the MFA library. Call once at startup.
@_cdecl("mfa_init")
public func mfa_init() -> Bool {
    // MTLContext.global initializes lazily; just verify GPU exists
    return MTLCreateSystemDefaultDevice() != nil
}

/// Preload cached pipelines into memory for zero-latency first call.
@_cdecl("mfa_preload_cache")
public func mfa_preload_cache() {
    // With monolithic IR, compilation is fast (~90ms) and cached in-memory.
    // No-op for now; could pre-register common configs if needed.
}

// MARK: - Kernel Creation (v1-v7)

@_cdecl("mfa_create_kernel")
public func mfa_create_kernel(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v2(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: false
    )
}

@_cdecl("mfa_create_kernel_v2")
public func mfa_create_kernel_v2(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool, use_bf16: Bool
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v3(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: use_bf16, window_size: 0
    )
}

@_cdecl("mfa_create_kernel_v3")
public func mfa_create_kernel_v3(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v7(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: use_bf16,
        window_size: window_size, quantized_kv: 0, bf16_backward: false,
        has_attn_bias: false, bias_batch_stride: 0, bias_head_stride: 0, bias_repeat_count: 0
    )
}

@_cdecl("mfa_create_kernel_v4")
public func mfa_create_kernel_v4(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32, quantized_kv: UInt16
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v7(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: use_bf16,
        window_size: window_size, quantized_kv: quantized_kv, bf16_backward: false,
        has_attn_bias: false, bias_batch_stride: 0, bias_head_stride: 0, bias_repeat_count: 0
    )
}

@_cdecl("mfa_create_kernel_v5")
public func mfa_create_kernel_v5(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32, quantized_kv: UInt16, bf16_backward: Bool
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v7(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: use_bf16,
        window_size: window_size, quantized_kv: quantized_kv, bf16_backward: bf16_backward,
        has_attn_bias: false, bias_batch_stride: 0, bias_head_stride: 0, bias_repeat_count: 0
    )
}

@_cdecl("mfa_create_kernel_v6")
public func mfa_create_kernel_v6(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32, quantized_kv: UInt16, bf16_backward: Bool,
    has_attn_bias: Bool, bias_batch_stride: UInt32, bias_head_stride: UInt32
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v7(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: use_bf16,
        window_size: window_size, quantized_kv: quantized_kv, bf16_backward: bf16_backward,
        has_attn_bias: has_attn_bias, bias_batch_stride: bias_batch_stride,
        bias_head_stride: bias_head_stride, bias_repeat_count: 0
    )
}

@_cdecl("mfa_create_kernel_v7")
public func mfa_create_kernel_v7(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32, quantized_kv: UInt16, bf16_backward: Bool,
    has_attn_bias: Bool, bias_batch_stride: UInt32, bias_head_stride: UInt32, bias_repeat_count: UInt32
) -> UnsafeMutableRawPointer? {
    let key = "\(seq_len_q)_\(seq_len_kv)_\(head_dim)_\(low_precision)_\(low_precision_outputs)_\(causal)_\(has_mask)_\(use_bf16)_\(window_size)_\(quantized_kv)_\(bf16_backward)_\(has_attn_bias)_\(bias_batch_stride)_\(bias_head_stride)_\(bias_repeat_count)"

    cacheLock.lock()
    defer { cacheLock.unlock() }

    if let cached = kernelCache[key] {
        return Unmanaged.passRetained(cached).toOpaque()
    }

    var quantizedKVPrecision: GEMMOperandPrecision? = nil
    if quantized_kv > 0 {
        quantizedKVPrecision = GEMMOperandPrecision(rawValue: quantized_kv)
    }

    do {
        let kernel = try MFAKernelCache(
            seqLenQ: Int(seq_len_q), seqLenKV: Int(seq_len_kv), headDim: Int(head_dim),
            lowPrecision: low_precision, lowPrecisionOutputs: low_precision_outputs,
            causal: causal, hasMask: has_mask, useBF16: use_bf16, windowSize: window_size,
            quantizedKV: quantizedKVPrecision, bf16Backward: bf16_backward,
            hasAttnBias: has_attn_bias, biasBatchStride: bias_batch_stride,
            biasHeadStride: bias_head_stride, biasRepeatCount: bias_repeat_count
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
}

// MARK: - Forward Encode (Batched Dispatch)

@_cdecl("mfa_forward_encode")
public func mfa_forward_encode(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, l_offset: Int64, mask_offset: Int64,
    batch_size: Int32, num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()
    let encoder: MTLComputeCommandEncoder = unsafeBitCast(encoder_ptr, to: MTLComputeCommandEncoder.self)

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)

    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    encoder.setComputePipelineState(cache.forwardPipeline)
    encoder.setThreadgroupMemoryLength(Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
        / Int(cache.forwardKernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads) * Int(batch_size), depth: 1)
    let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    return true
}

/// Forward with attention bias
@_cdecl("mfa_forward_encode_bias")
public func mfa_forward_encode_bias(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    attn_bias_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, l_offset: Int64, mask_offset: Int64, attn_bias_offset: Int64,
    batch_size: Int32, num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()
    let encoder: MTLComputeCommandEncoder = unsafeBitCast(encoder_ptr, to: MTLComputeCommandEncoder.self)

    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)

    var maskBuffer: MTLBuffer? = nil
    if let mask_ptr = mask_ptr {
        maskBuffer = unsafeBitCast(mask_ptr, to: MTLBuffer.self)
    }

    var attnBiasBuffer: MTLBuffer? = nil
    if let attn_bias_ptr = attn_bias_ptr {
        attnBiasBuffer = unsafeBitCast(attn_bias_ptr, to: MTLBuffer.self)
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    encoder.setComputePipelineState(cache.forwardPipeline)
    encoder.setThreadgroupMemoryLength(Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    if let attnBiasBuffer = attnBiasBuffer {
        encoder.setBuffer(attnBiasBuffer, offset: Int(attn_bias_offset), index: 11)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
        / Int(cache.forwardKernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads) * Int(batch_size), depth: 1)
    let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    return true
}

/// Forward with quantized K/V
@_cdecl("mfa_forward_encode_quantized")
public func mfa_forward_encode_quantized(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    k_scale_ptr: UnsafeMutableRawPointer,
    v_scale_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, l_offset: Int64,
    k_scale_offset: Int64, v_scale_offset: Int64, mask_offset: Int64,
    batch_size: Int32, num_heads: Int32
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
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    encoder.setComputePipelineState(cache.forwardPipeline)
    encoder.setThreadgroupMemoryLength(Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder.setBuffer(kScaleBuffer, offset: Int(k_scale_offset), index: 20)
    encoder.setBuffer(vScaleBuffer, offset: Int(v_scale_offset), index: 21)
    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
        / Int(cache.forwardKernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads) * Int(batch_size), depth: 1)
    let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    return true
}

// MARK: - Legacy Forward (own command buffer)

@_cdecl("mfa_forward")
public func mfa_forward(
    kernel_handle: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,
    mask_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, l_offset: Int64, mask_offset: Int64,
    batch_size: Int32, num_heads: Int32
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

    let commandQueue = MTLContext.global.commandQueue
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    encoder.setComputePipelineState(cache.forwardPipeline)
    encoder.setThreadgroupMemoryLength(Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount = (seqLenQ + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
        / Int(cache.forwardKernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads) * Int(batch_size), depth: 1)
    let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.error == nil
}

// MARK: - Backward Encode (Batched Dispatch)

@_cdecl("mfa_backward_encode")
public func mfa_backward_encode(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,
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
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, do_offset: Int64, l_offset: Int64, d_offset: Int64,
    dq_offset: Int64, dk_offset: Int64, dv_offset: Int64, mask_offset: Int64,
    batch_size: Int32, num_heads: Int32
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
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    // Phase 1: Backward Query
    encoder.setComputePipelineState(bwdQPipeline)
    encoder.setThreadgroupMemoryLength(Int(bwdQKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)
    encoder.setBuffer(dBuffer, offset: Int(d_offset), index: 5)
    encoder.setBuffer(doBuffer, offset: Int(do_offset), index: 6)
    encoder.setBuffer(dqBuffer, offset: Int(dq_offset), index: 9)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount1 = (seqLenQ + Int(bwdQKernel.blockDimensions.parallelization) - 1)
        / Int(bwdQKernel.blockDimensions.parallelization)
    encoder.dispatchThreadgroups(
        MTLSize(width: blockCount1, height: Int(num_heads) * Int(batch_size), depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(bwdQKernel.threadgroupSize), height: 1, depth: 1)
    )

    // Phase 2: Backward KV
    encoder.setComputePipelineState(bwdKVPipeline)
    encoder.setThreadgroupMemoryLength(Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)
    encoder.setBuffer(dBuffer, offset: Int(d_offset), index: 5)
    encoder.setBuffer(doBuffer, offset: Int(do_offset), index: 6)
    encoder.setBuffer(dvBuffer, offset: Int(dv_offset), index: 7)
    encoder.setBuffer(dkBuffer, offset: Int(dk_offset), index: 8)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount2 = (seqLenKV + Int(bwdKVKernel.blockDimensions.parallelization) - 1)
        / Int(bwdKVKernel.blockDimensions.parallelization)
    encoder.dispatchThreadgroups(
        MTLSize(width: blockCount2, height: Int(num_heads) * Int(batch_size), depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(bwdKVKernel.threadgroupSize), height: 1, depth: 1)
    )

    return true
}

// MARK: - Backward with Bias (PyTorch encoder)

@_cdecl("mfa_backward_encode_bias")
public func mfa_backward_encode_bias(
    kernel_handle: UnsafeMutableRawPointer,
    encoder_ptr: UnsafeMutableRawPointer,
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
    bias_ptr: UnsafeMutableRawPointer?,
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, do_offset: Int64, l_offset: Int64, d_offset: Int64,
    dq_offset: Int64, dk_offset: Int64, dv_offset: Int64, mask_offset: Int64,
    bias_offset: Int64,
    batch_size: Int32, num_heads: Int32
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

    var biasBuffer: MTLBuffer? = nil
    if let bias_ptr = bias_ptr {
        biasBuffer = unsafeBitCast(bias_ptr, to: MTLBuffer.self)
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    // Phase 1: Backward Query
    encoder.setComputePipelineState(bwdQPipeline)
    encoder.setThreadgroupMemoryLength(Int(bwdQKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)
    encoder.setBuffer(dBuffer, offset: Int(d_offset), index: 5)
    encoder.setBuffer(doBuffer, offset: Int(do_offset), index: 6)
    encoder.setBuffer(dqBuffer, offset: Int(dq_offset), index: 9)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    if let biasBuffer = biasBuffer {
        encoder.setBuffer(biasBuffer, offset: Int(bias_offset), index: 11)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount1 = (seqLenQ + Int(bwdQKernel.blockDimensions.parallelization) - 1)
        / Int(bwdQKernel.blockDimensions.parallelization)
    encoder.dispatchThreadgroups(
        MTLSize(width: blockCount1, height: Int(num_heads) * Int(batch_size), depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(bwdQKernel.threadgroupSize), height: 1, depth: 1)
    )

    // Phase 2: Backward KV
    encoder.setComputePipelineState(bwdKVPipeline)
    encoder.setThreadgroupMemoryLength(Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder.setBuffer(lBuffer, offset: Int(l_offset), index: 4)
    encoder.setBuffer(dBuffer, offset: Int(d_offset), index: 5)
    encoder.setBuffer(doBuffer, offset: Int(do_offset), index: 6)
    encoder.setBuffer(dvBuffer, offset: Int(dv_offset), index: 7)
    encoder.setBuffer(dkBuffer, offset: Int(dk_offset), index: 8)

    if let maskBuffer = maskBuffer {
        encoder.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    if let biasBuffer = biasBuffer {
        encoder.setBuffer(biasBuffer, offset: Int(bias_offset), index: 11)
    }

    encoder.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount2 = (seqLenKV + Int(bwdKVKernel.blockDimensions.parallelization) - 1)
        / Int(bwdKVKernel.blockDimensions.parallelization)
    encoder.dispatchThreadgroups(
        MTLSize(width: blockCount2, height: Int(num_heads) * Int(batch_size), depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(bwdKVKernel.threadgroupSize), height: 1, depth: 1)
    )

    return true
}

// MARK: - Legacy Backward (own command buffer)

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
    q_offset: Int64, k_offset: Int64, v_offset: Int64,
    o_offset: Int64, do_offset: Int64, l_offset: Int64, d_offset: Int64,
    dq_offset: Int64, dk_offset: Int64, dv_offset: Int64, mask_offset: Int64,
    batch_size: Int32, num_heads: Int32
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

    let commandQueue = MTLContext.global.commandQueue
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let paramsBuffer = cache.createBatchedParamsBuffer(numHeads: Int(num_heads))

    // Phase 1: Backward Query
    guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else { return false }

    encoder1.setComputePipelineState(bwdQPipeline)
    encoder1.setThreadgroupMemoryLength(Int(bwdQKernel.threadgroupMemoryAllocation), index: 0)

    encoder1.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder1.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder1.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder1.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder1.setBuffer(lBuffer, offset: Int(l_offset), index: 4)
    encoder1.setBuffer(dBuffer, offset: Int(d_offset), index: 5)
    encoder1.setBuffer(doBuffer, offset: Int(do_offset), index: 6)
    encoder1.setBuffer(dqBuffer, offset: Int(dq_offset), index: 9)

    if let maskBuffer = maskBuffer {
        encoder1.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder1.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount1 = (seqLenQ + Int(bwdQKernel.blockDimensions.parallelization) - 1)
        / Int(bwdQKernel.blockDimensions.parallelization)
    encoder1.dispatchThreadgroups(
        MTLSize(width: blockCount1, height: Int(num_heads) * Int(batch_size), depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(bwdQKernel.threadgroupSize), height: 1, depth: 1)
    )
    encoder1.endEncoding()

    // Phase 2: Backward KV
    guard let encoder2 = commandBuffer.makeComputeCommandEncoder() else { return false }

    encoder2.setComputePipelineState(bwdKVPipeline)
    encoder2.setThreadgroupMemoryLength(Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)

    encoder2.setBuffer(qBuffer, offset: Int(q_offset), index: 0)
    encoder2.setBuffer(kBuffer, offset: Int(k_offset), index: 1)
    encoder2.setBuffer(vBuffer, offset: Int(v_offset), index: 2)
    encoder2.setBuffer(oBuffer, offset: Int(o_offset), index: 3)
    encoder2.setBuffer(lBuffer, offset: Int(l_offset), index: 4)
    encoder2.setBuffer(dBuffer, offset: Int(d_offset), index: 5)
    encoder2.setBuffer(doBuffer, offset: Int(do_offset), index: 6)
    encoder2.setBuffer(dvBuffer, offset: Int(dv_offset), index: 7)
    encoder2.setBuffer(dkBuffer, offset: Int(dk_offset), index: 8)

    if let maskBuffer = maskBuffer {
        encoder2.setBuffer(maskBuffer, offset: Int(mask_offset), index: 10)
    }

    encoder2.setBuffer(paramsBuffer, offset: 0, index: 30)

    let blockCount2 = (seqLenKV + Int(bwdKVKernel.blockDimensions.parallelization) - 1)
        / Int(bwdKVKernel.blockDimensions.parallelization)
    encoder2.dispatchThreadgroups(
        MTLSize(width: blockCount2, height: Int(num_heads) * Int(batch_size), depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(bwdKVKernel.threadgroupSize), height: 1, depth: 1)
    )
    encoder2.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.error == nil
}

// MARK: - Utility Functions

@_cdecl("mfa_release_kernel")
public func mfa_release_kernel(kernel_handle: UnsafeMutableRawPointer) {
    Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).release()
}

@_cdecl("mfa_version")
public func mfa_version() -> UnsafePointer<CChar>? {
    let version = "0.5.0-monolithic"
    return (version as NSString).utf8String
}

@_cdecl("mfa_precompile")
public func mfa_precompile() {
    // With monolithic IR + MetalASM, compilation is fast (~90ms) and cached in-memory.
    // Pre-registration can be done via AttentionKernel.register() if needed.
}

@_cdecl("mfa_clear_cache")
public func mfa_clear_cache() {
    cacheLock.lock()
    kernelCache.removeAll()
    AttentionKernel.pipelineCache.removeAll()
    cacheLock.unlock()
}

@_cdecl("mfa_set_kernels_dir")
public func mfa_set_kernels_dir(path: UnsafePointer<CChar>) {
    // No-op: monolithic IR doesn't use shipped metallib files
}
