/**
 * MFA Bridge - C-callable API for metal-flash-attention
 *
 * This module provides PyTorch-friendly C functions to access Flash Attention on Metal.
 *
 * BATCHED DISPATCH: All encode functions use 3D dispatch (blocks x heads x batch)
 * instead of nested loops. This dramatically improves performance for high-batch
 * workloads (e.g., 484 batches goes from 2904 dispatches to 1 dispatch).
 */

import Foundation
import Metal
import FlashAttention

// MARK: - Batched Dispatch Parameters

/// Struct matching the Metal BatchedParams struct for uniform buffer
/// This is passed to the kernel at buffer index 30
struct BatchedParams {
    var num_heads: UInt32
    var Q_head_stride: UInt32
    var K_head_stride: UInt32
    var V_head_stride: UInt32
    var O_head_stride: UInt32
    var L_head_stride: UInt32
    var mask_head_stride: UInt32
}

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

    // Reusable buffer for batched params (per-cache to avoid contention)
    var batchedParamsBuffer: MTLBuffer?

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
        biasRepeatCount: UInt32 = 0,
        device: MTLDevice
    ) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

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

        // Create forward kernel
        let fwdKernelDesc = desc.kernelDescriptor(type: .forward)
        self.forwardKernel = AttentionKernel(descriptor: fwdKernelDesc)
        let fwdSource = forwardKernel.createSource()

        let fwdConstants = MTLFunctionConstantValues()
        desc.setFunctionConstants(fwdConstants)

        let dims = desc.matrixDimensions!
        self.forwardPipeline = try MetallibCache.shared.getPipeline(
            source: fwdSource,
            functionName: "attention",
            constants: fwdConstants,
            rowDim: dims.row,
            colDim: dims.column,
            device: device
        )

        // Pre-allocate batched params buffer
        self.batchedParamsBuffer = device.makeBuffer(
            length: MemoryLayout<BatchedParams>.size,
            options: .storageModeShared
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

    /// Get strides for batched dispatch (in ELEMENTS, not bytes)
    func getStrides() -> (qStride: UInt32, kStride: UInt32, oStride: UInt32, lStride: UInt32, maskStride: UInt32) {
        let seqLenQ = UInt32(descriptor.matrixDimensions!.row)
        let seqLenKV = UInt32(descriptor.matrixDimensions!.column)
        let headDim = UInt32(descriptor.matrixDimensions!.head)

        // For NF4 quantization, K/V have packed head dimension (D/2 bytes per position)
        // So the stride needs to use headDim/2 for K/V
        let isNF4 = (descriptor.quantizedKV == .NF4)
        let kvHeadDim = isNF4 ? (headDim / 2) : headDim

        return (
            qStride: seqLenQ * headDim,
            kStride: seqLenKV * kvHeadDim,
            oStride: seqLenQ * headDim,
            lStride: seqLenQ,
            maskStride: seqLenQ * seqLenKV
        )
    }

    /// Setup batched params buffer for dispatch
    func setupBatchedParams(numHeads: Int) -> MTLBuffer {
        let strides = getStrides()
        var params = BatchedParams(
            num_heads: UInt32(numHeads),
            Q_head_stride: strides.qStride,
            K_head_stride: strides.kStride,
            V_head_stride: strides.kStride,
            O_head_stride: strides.oStride,
            L_head_stride: strides.lStride,
            mask_head_stride: strides.maskStride
        )
        memcpy(batchedParamsBuffer!.contents(), &params, MemoryLayout<BatchedParams>.size)
        return batchedParamsBuffer!
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
@_cdecl("mfa_init")
public func mfa_init() -> Bool {
    globalDevice = MTLCreateSystemDefaultDevice()
    guard globalDevice != nil else { return false }
    globalQueue = globalDevice!.makeCommandQueue()
    return globalQueue != nil
}

/// Preload cached pipelines into memory for zero-latency first call.
@_cdecl("mfa_preload_cache")
public func mfa_preload_cache() {
    guard let device = globalDevice ?? MTLCreateSystemDefaultDevice() else { return }
    if globalDevice == nil {
        globalDevice = device
        globalQueue = device.makeCommandQueue()
    }

    let seqLens = [512, 1024, 2048, 4096, 8192]
    let headDims = [64, 128]
    var configs: [(seqLen: Int, headDim: Int, lowPrecision: Bool)] = []

    for seqLen in seqLens {
        for headDim in headDims {
            configs.append((seqLen, headDim, true))
        }
    }

    MetallibCache.shared.preloadConfigurations(device: device, configurations: configs)
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
    let key = cacheKey(Int(seq_len_q), Int(seq_len_kv), Int(head_dim), low_precision, low_precision_outputs, causal, has_mask, use_bf16, window_size)

    cacheLock.lock()
    defer { cacheLock.unlock() }

    if let cached = kernelCache[key] {
        return Unmanaged.passRetained(cached).toOpaque()
    }

    do {
        let kernel = try MFAKernelCache(
            seqLenQ: Int(seq_len_q), seqLenKV: Int(seq_len_kv), headDim: Int(head_dim),
            lowPrecision: low_precision, lowPrecisionOutputs: low_precision_outputs,
            causal: causal, hasMask: has_mask, useBF16: use_bf16, windowSize: window_size,
            device: getDevice()
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
}

@_cdecl("mfa_create_kernel_v4")
public func mfa_create_kernel_v4(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32, quantized_kv: UInt16
) -> UnsafeMutableRawPointer? {
    return mfa_create_kernel_v5(
        seq_len_q: seq_len_q, seq_len_kv: seq_len_kv, head_dim: head_dim,
        low_precision: low_precision, low_precision_outputs: low_precision_outputs,
        causal: causal, has_mask: has_mask, use_bf16: use_bf16,
        window_size: window_size, quantized_kv: quantized_kv, bf16_backward: false
    )
}

@_cdecl("mfa_create_kernel_v5")
public func mfa_create_kernel_v5(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, low_precision_outputs: Bool, causal: Bool, has_mask: Bool,
    use_bf16: Bool, window_size: UInt32, quantized_kv: UInt16, bf16_backward: Bool
) -> UnsafeMutableRawPointer? {
    let key = "\(seq_len_q)_\(seq_len_kv)_\(head_dim)_\(low_precision)_\(low_precision_outputs)_\(causal)_\(has_mask)_\(use_bf16)_\(window_size)_\(quantized_kv)_\(bf16_backward)"

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
            device: getDevice()
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
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
            biasHeadStride: bias_head_stride, biasRepeatCount: bias_repeat_count,
            device: getDevice()
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
}

// MARK: - Forward Encode (Batched Dispatch)

/// Execute forward attention pass using PyTorch's compute command encoder.
/// Uses batched 3D dispatch for maximum performance.
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
    let paramsBuffer = cache.setupBatchedParams(numHeads: Int(num_heads))

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
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads), depth: Int(batch_size))
    let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    return true
}

/// Forward with attention bias - uses batched dispatch
/// Note: Attention bias with biasRepeatCount still needs per-batch handling for modulo logic
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
    let paramsBuffer = cache.setupBatchedParams(numHeads: Int(num_heads))

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
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads), depth: Int(batch_size))
    let groupSize = MTLSize(width: Int(cache.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    return true
}

/// Forward with quantized K/V - uses batched dispatch
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
    let paramsBuffer = cache.setupBatchedParams(numHeads: Int(num_heads))

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
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads), depth: Int(batch_size))
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

    guard let commandBuffer = cache.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let paramsBuffer = cache.setupBatchedParams(numHeads: Int(num_heads))

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
    let gridSize = MTLSize(width: blockCount, height: Int(num_heads), depth: Int(batch_size))
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
    let paramsBuffer = cache.setupBatchedParams(numHeads: Int(num_heads))

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
        MTLSize(width: blockCount1, height: Int(num_heads), depth: Int(batch_size)),
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
        MTLSize(width: blockCount2, height: Int(num_heads), depth: Int(batch_size)),
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

    guard let commandBuffer = cache.commandQueue.makeCommandBuffer() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let paramsBuffer = cache.setupBatchedParams(numHeads: Int(num_heads))

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
        MTLSize(width: blockCount1, height: Int(num_heads), depth: Int(batch_size)),
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
        MTLSize(width: blockCount2, height: Int(num_heads), depth: Int(batch_size)),
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
    let version = "0.2.0-batched"
    return (version as NSString).utf8String
}

@_cdecl("mfa_precompile")
public func mfa_precompile() {
    let device = getDevice()
    var configs: [(seqLen: Int, headDim: Int, lowPrecision: Bool)] = []

    let seqLens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    let headDims = [32, 48, 64, 80, 96, 128]

    for seqLen in seqLens {
        for headDim in headDims {
            configs.append((seqLen, headDim, false))
            configs.append((seqLen, headDim, true))
        }
    }

    MetallibCache.shared.precompile(device: device, configurations: configs)
}

@_cdecl("mfa_clear_cache")
public func mfa_clear_cache() {
    MetallibCache.shared.clearCache()
}

@_cdecl("mfa_set_kernels_dir")
public func mfa_set_kernels_dir(path: UnsafePointer<CChar>) {
    let pathString = String(cString: path)
    MetallibCache.shared.setShippedKernelsDir(pathString)
}

// Debug function to check if kernel has bias code
@_cdecl("mfa_debug_kernel_source")
public func mfa_debug_kernel_source(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool, has_attn_bias: Bool
) -> UnsafePointer<CChar>? {
    var desc = AttentionDescriptor()
    desc.lowPrecisionInputs = low_precision
    desc.lowPrecisionIntermediates = false
    desc.lowPrecisionOutputs = low_precision
    desc.hasAttnBias = has_attn_bias
    desc.biasHeadStride = UInt32(seq_len_q * seq_len_kv)
    desc.matrixDimensions = (row: UInt32(seq_len_q), column: UInt32(seq_len_kv), head: UInt16(head_dim))
    desc.transposeState = (Q: false, K: false, V: false, O: false)
    
    let kernelDesc = desc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)
    let source = kernel.createSource()
    
    // Check for bias code
    var result = ""
    if source.contains("attn_bias") {
        result += "✅ Contains attn_bias\n"
    } else {
        result += "❌ NO attn_bias found\n"
    }
    if source.contains("Add attention bias") {
        result += "✅ Has addAttnBias() code\n"
    } else {
        result += "❌ NO addAttnBias() code\n"
    }
    if source.contains("[[buffer(11)]]") {
        result += "✅ Buffer 11 bound\n"
    } else {
        result += "❌ Buffer 11 NOT bound\n"
    }
    
    // Print first 200 chars of source for debugging
    result += "\nFirst 500 chars of source:\n"
    result += String(source.prefix(500))
    
    return (result as NSString).utf8String
}

// Debug function to dump full kernel source with bias code
@_cdecl("mfa_dump_kernel_bias_code")
public func mfa_dump_kernel_bias_code(
    seq_len_q: Int32, seq_len_kv: Int32, head_dim: Int32,
    low_precision: Bool
) -> UnsafePointer<CChar>? {
    var desc = AttentionDescriptor()
    desc.lowPrecisionInputs = low_precision
    desc.lowPrecisionIntermediates = false
    desc.lowPrecisionOutputs = low_precision
    desc.hasAttnBias = true
    desc.biasHeadStride = UInt32(seq_len_q * seq_len_kv)
    desc.matrixDimensions = (row: UInt32(seq_len_q), column: UInt32(seq_len_kv), head: UInt16(head_dim))
    desc.transposeState = (Q: false, K: false, V: false, O: false)
    
    let kernelDesc = desc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)
    let source = kernel.createSource()
    
    // Extract just the bias-related code
    var result = ""
    let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
    var inBiasSection = false
    var braceCount = 0
    
    for (i, line) in lines.enumerated() {
        let lineStr = String(line)
        if lineStr.contains("Add attention bias") {
            inBiasSection = true
            result += "Line \(i): \(lineStr)\n"
        } else if inBiasSection {
            result += "Line \(i): \(lineStr)\n"
            // Track braces to know when section ends
            braceCount += lineStr.filter { $0 == "{" }.count
            braceCount -= lineStr.filter { $0 == "}" }.count
            if braceCount <= 0 && lineStr.contains("}") {
                inBiasSection = false
                break
            }
        }
        
        // Also capture buffer declarations
        if lineStr.contains("attn_bias") && lineStr.contains("buffer") {
            result += "BUFFER: \(lineStr)\n"
        }
    }
    
    if result.isEmpty {
        result = "NO BIAS CODE FOUND IN KERNEL!"
    }
    
    return (result as NSString).utf8String
}
