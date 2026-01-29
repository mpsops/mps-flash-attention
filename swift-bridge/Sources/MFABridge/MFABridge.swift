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
        device: MTLDevice
    ) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Configure descriptor
        var desc = AttentionDescriptor()
        desc.lowPrecisionInputs = lowPrecision
        desc.lowPrecisionIntermediates = lowPrecision
        desc.lowPrecisionOutputs = lowPrecisionOutputs
        desc.causal = causal
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

private func cacheKey(_ seqQ: Int, _ seqKV: Int, _ headDim: Int, _ lowPrec: Bool, _ lowPrecOut: Bool, _ causal: Bool) -> String {
    "\(seqQ)_\(seqKV)_\(headDim)_\(lowPrec)_\(lowPrecOut)_\(causal)"
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
/// low_precision_outputs: if true, output O is FP16 (saves memory, avoids fp32->fp16 copy)
/// causal: if true, applies causal masking (lower triangular attention)
@_cdecl("mfa_create_kernel")
public func mfa_create_kernel(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    low_precision: Bool,
    low_precision_outputs: Bool,
    causal: Bool
) -> UnsafeMutableRawPointer? {
    let key = cacheKey(Int(seq_len_q), Int(seq_len_kv), Int(head_dim), low_precision, low_precision_outputs, causal)

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
            device: getDevice()
        )
        kernelCache[key] = kernel
        return Unmanaged.passRetained(kernel).toOpaque()
    } catch {
        print("MFA Error creating kernel: \(error)")
        return nil
    }
}

/// Execute forward attention pass with storage byte offsets.
/// Buffers are id<MTLBuffer> passed as raw pointers from C++.
/// Offsets are byte offsets into each buffer where tensor data starts.
@_cdecl("mfa_forward")
public func mfa_forward(
    kernel_handle: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,  // id<MTLBuffer> cast to void*
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    l_ptr: UnsafeMutableRawPointer,  // logsumexp buffer
    q_offset: Int64,   // byte offset into q_ptr buffer
    k_offset: Int64,   // byte offset into k_ptr buffer
    v_offset: Int64,   // byte offset into v_ptr buffer
    o_offset: Int64,   // byte offset into o_ptr buffer
    l_offset: Int64,   // byte offset into l_ptr buffer
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()

    // id<MTLBuffer> is an Objective-C object pointer, so we use unsafeBitCast
    // from the raw pointer passed from C++
    let qBuffer: MTLBuffer = unsafeBitCast(q_ptr, to: MTLBuffer.self)
    let kBuffer: MTLBuffer = unsafeBitCast(k_ptr, to: MTLBuffer.self)
    let vBuffer: MTLBuffer = unsafeBitCast(v_ptr, to: MTLBuffer.self)
    let oBuffer: MTLBuffer = unsafeBitCast(o_ptr, to: MTLBuffer.self)
    let lBuffer: MTLBuffer = unsafeBitCast(l_ptr, to: MTLBuffer.self)

    guard let commandBuffer = cache.commandQueue.makeCommandBuffer() else {
        return false
    }

    let seqLen = Int(cache.descriptor.matrixDimensions!.row)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)
    let inputElementSize = cache.descriptor.lowPrecisionInputs ? 2 : 4  // fp16 vs fp32
    // Output precision depends on lowPrecisionOutputs flag
    let outputElementSize = cache.descriptor.lowPrecisionOutputs ? 2 : 4  // fp16 vs fp32

    // Process each batch and head
    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            // Per-head offset within the tensor (after storage offset)
            let inputHeadOffset = (b * Int(num_heads) + h) * seqLen * headDim * inputElementSize
            let outputHeadOffset = (b * Int(num_heads) + h) * seqLen * headDim * outputElementSize
            let lHeadOffset = (b * Int(num_heads) + h) * seqLen * 4  // L is always fp32

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return false
            }

            encoder.setComputePipelineState(cache.forwardPipeline)
            encoder.setThreadgroupMemoryLength(
                Int(cache.forwardKernel.threadgroupMemoryAllocation), index: 0)

            // Buffer bindings: Q=0, K=1, V=2, O=3, L=4
            // Total offset = storage_offset + per_head_offset
            encoder.setBuffer(qBuffer, offset: Int(q_offset) + inputHeadOffset, index: 0)
            encoder.setBuffer(kBuffer, offset: Int(k_offset) + inputHeadOffset, index: 1)
            encoder.setBuffer(vBuffer, offset: Int(v_offset) + inputHeadOffset, index: 2)
            encoder.setBuffer(oBuffer, offset: Int(o_offset) + outputHeadOffset, index: 3)
            encoder.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)

            // Dispatch
            let blockCount = (seqLen + Int(cache.forwardKernel.blockDimensions.parallelization) - 1)
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

/// Execute backward attention pass (computes dQ, dK, dV from dO).
/// This runs both backwardQuery and backwardKeyValue kernels.
/// Buffer bindings: Q=0, K=1, V=2, O=3, L=4, D=5, dO=6, dV=7, dK=8, dQ=9
@_cdecl("mfa_backward")
public func mfa_backward(
    kernel_handle: UnsafeMutableRawPointer,
    q_ptr: UnsafeMutableRawPointer,
    k_ptr: UnsafeMutableRawPointer,
    v_ptr: UnsafeMutableRawPointer,
    o_ptr: UnsafeMutableRawPointer,
    do_ptr: UnsafeMutableRawPointer,  // gradient of output
    l_ptr: UnsafeMutableRawPointer,   // logsumexp from forward
    d_ptr: UnsafeMutableRawPointer,   // D buffer (dO * O reduction)
    dq_ptr: UnsafeMutableRawPointer,  // gradient of Q (output)
    dk_ptr: UnsafeMutableRawPointer,  // gradient of K (output)
    dv_ptr: UnsafeMutableRawPointer,  // gradient of V (output)
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
    batch_size: Int32,
    num_heads: Int32
) -> Bool {
    let cache = Unmanaged<MFAKernelCache>.fromOpaque(kernel_handle).takeUnretainedValue()

    // Ensure backward kernels are created
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

    // Cast buffers
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

    guard let commandBuffer = cache.commandQueue.makeCommandBuffer() else {
        return false
    }

    let seqLenQ = Int(cache.descriptor.matrixDimensions!.row)
    let seqLenKV = Int(cache.descriptor.matrixDimensions!.column)
    let headDim = Int(cache.descriptor.matrixDimensions!.head)
    let inputElementSize = cache.descriptor.lowPrecisionInputs ? 2 : 4
    let outputElementSize = cache.descriptor.lowPrecisionOutputs ? 2 : 4

    // Process each batch and head
    for b in 0..<Int(batch_size) {
        for h in 0..<Int(num_heads) {
            let bhIndex = b * Int(num_heads) + h

            // Offsets for Q, K, V (input precision)
            let qkvHeadOffset = bhIndex * seqLenQ * headDim * inputElementSize
            let kvHeadOffset = bhIndex * seqLenKV * headDim * inputElementSize

            // Offsets for O, dO (output precision)
            let oHeadOffset = bhIndex * seqLenQ * headDim * outputElementSize

            // Offsets for L, D (always fp32)
            let lHeadOffset = bhIndex * seqLenQ * 4
            let dHeadOffset = bhIndex * seqLenQ * 4

            // Offsets for gradients dQ, dK, dV (same as inputs - fp32 always for gradients)
            let dqHeadOffset = bhIndex * seqLenQ * headDim * 4  // dQ always fp32
            let dkvHeadOffset = bhIndex * seqLenKV * headDim * 4  // dK, dV always fp32

            // === Phase 1: Backward Query (computes D and dQ) ===
            guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else {
                return false
            }

            encoder1.setComputePipelineState(bwdQPipeline)
            encoder1.setThreadgroupMemoryLength(Int(bwdQKernel.threadgroupMemoryAllocation), index: 0)

            // Buffer bindings for backwardQuery: Q=0, K=1, V=2, O=3, L=4, D=5, dO=6, dQ=9
            encoder1.setBuffer(qBuffer, offset: Int(q_offset) + qkvHeadOffset, index: 0)
            encoder1.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder1.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder1.setBuffer(oBuffer, offset: Int(o_offset) + oHeadOffset, index: 3)
            encoder1.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)
            encoder1.setBuffer(dBuffer, offset: Int(d_offset) + dHeadOffset, index: 5)
            encoder1.setBuffer(doBuffer, offset: Int(do_offset) + oHeadOffset, index: 6)
            encoder1.setBuffer(dqBuffer, offset: Int(dq_offset) + dqHeadOffset, index: 9)

            let blockCount1 = (seqLenQ + Int(bwdQKernel.blockDimensions.parallelization) - 1)
                / Int(bwdQKernel.blockDimensions.parallelization)
            encoder1.dispatchThreadgroups(
                MTLSize(width: blockCount1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Int(bwdQKernel.threadgroupSize), height: 1, depth: 1)
            )
            encoder1.endEncoding()

            // === Phase 2: Backward KV (computes dK and dV) ===
            guard let encoder2 = commandBuffer.makeComputeCommandEncoder() else {
                return false
            }

            encoder2.setComputePipelineState(bwdKVPipeline)
            encoder2.setThreadgroupMemoryLength(Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)

            // Buffer bindings for backwardKeyValue: Q=0, K=1, V=2, O=3, L=4, D=5, dO=6, dV=7, dK=8
            encoder2.setBuffer(qBuffer, offset: Int(q_offset) + qkvHeadOffset, index: 0)
            encoder2.setBuffer(kBuffer, offset: Int(k_offset) + kvHeadOffset, index: 1)
            encoder2.setBuffer(vBuffer, offset: Int(v_offset) + kvHeadOffset, index: 2)
            encoder2.setBuffer(oBuffer, offset: Int(o_offset) + oHeadOffset, index: 3)
            encoder2.setBuffer(lBuffer, offset: Int(l_offset) + lHeadOffset, index: 4)
            encoder2.setBuffer(dBuffer, offset: Int(d_offset) + dHeadOffset, index: 5)
            encoder2.setBuffer(doBuffer, offset: Int(do_offset) + oHeadOffset, index: 6)
            encoder2.setBuffer(dvBuffer, offset: Int(dv_offset) + dkvHeadOffset, index: 7)
            encoder2.setBuffer(dkBuffer, offset: Int(dk_offset) + dkvHeadOffset, index: 8)

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
