/**
 * Swift wrapper for metal-flash-attention
 * Exposes C-callable API via @_cdecl for PyTorch integration
 */

import Foundation
import Metal

// MARK: - Kernel Cache

/// Cached kernel with compiled pipeline
final class CachedAttentionKernel {
    let forwardKernel: AttentionKernel
    let forwardPipeline: MTLComputePipelineState
    let backwardQueryKernel: AttentionKernel?
    let backwardQueryPipeline: MTLComputePipelineState?
    let backwardKVKernel: AttentionKernel?
    let backwardKVPipeline: MTLComputePipelineState?
    let descriptor: AttentionDescriptor

    init(descriptor: AttentionDescriptor, device: MTLDevice, includeBackward: Bool) throws {
        self.descriptor = descriptor

        // Forward kernel
        let forwardDesc = descriptor.kernelDescriptor(type: .forward)
        self.forwardKernel = AttentionKernel(descriptor: forwardDesc)
        self.forwardPipeline = try Self.compilePipeline(
            kernel: forwardKernel, descriptor: descriptor, device: device)

        if includeBackward {
            // Backward query kernel
            let bwdQueryDesc = descriptor.kernelDescriptor(type: .backwardQuery)
            let bwdQueryKernel = AttentionKernel(descriptor: bwdQueryDesc)
            self.backwardQueryKernel = bwdQueryKernel
            self.backwardQueryPipeline = try Self.compilePipeline(
                kernel: bwdQueryKernel, descriptor: descriptor, device: device)

            // Backward KV kernel
            let bwdKVDesc = descriptor.kernelDescriptor(type: .backwardKeyValue)
            let bwdKVKernel = AttentionKernel(descriptor: bwdKVDesc)
            self.backwardKVKernel = bwdKVKernel
            self.backwardKVPipeline = try Self.compilePipeline(
                kernel: bwdKVKernel, descriptor: descriptor, device: device)
        } else {
            self.backwardQueryKernel = nil
            self.backwardQueryPipeline = nil
            self.backwardKVKernel = nil
            self.backwardKVPipeline = nil
        }
    }

    private static func compilePipeline(
        kernel: AttentionKernel,
        descriptor: AttentionDescriptor,
        device: MTLDevice
    ) throws -> MTLComputePipelineState {
        let source = kernel.createSource()
        let library = try device.makeLibrary(source: source, options: nil)

        let constants = MTLFunctionConstantValues()
        descriptor.setFunctionConstants(constants)

        let function = try library.makeFunction(name: "attention", constantValues: constants)

        let pipelineDesc = MTLComputePipelineDescriptor()
        pipelineDesc.computeFunction = function
        pipelineDesc.maxTotalThreadsPerThreadgroup = 1024

        return try device.makeComputePipelineState(
            descriptor: pipelineDesc, options: [], reflection: nil)
    }
}

/// Global kernel cache
private var kernelCache: [String: CachedAttentionKernel] = [:]
private let cacheLock = NSLock()

func cacheKey(seqLenQ: Int, seqLenKV: Int, headDim: Int, isCausal: Bool, lowPrecision: Bool) -> String {
    return "\(seqLenQ)_\(seqLenKV)_\(headDim)_\(isCausal)_\(lowPrecision)"
}

// MARK: - C-Callable API

/// Opaque handle for cached kernel
public typealias MFAKernelHandle = UnsafeMutableRawPointer

/// Create or retrieve cached attention kernel
@_cdecl("mfa_create_attention_kernel")
public func mfa_create_attention_kernel(
    seq_len_q: Int32,
    seq_len_kv: Int32,
    head_dim: Int32,
    is_causal: Bool,
    low_precision: Bool,
    include_backward: Bool
) -> MFAKernelHandle? {
    let key = cacheKey(
        seqLenQ: Int(seq_len_q),
        seqLenKV: Int(seq_len_kv),
        headDim: Int(head_dim),
        isCausal: is_causal,
        lowPrecision: low_precision
    )

    cacheLock.lock()
    defer { cacheLock.unlock() }

    if let cached = kernelCache[key] {
        return Unmanaged.passRetained(cached).toOpaque()
    }

    // Create new descriptor
    var desc = AttentionDescriptor()
    desc.lowPrecisionInputs = low_precision
    desc.lowPrecisionIntermediates = low_precision
    desc.matrixDimensions = (
        row: UInt32(seq_len_q),
        column: UInt32(seq_len_kv),
        head: UInt16(head_dim)
    )
    desc.transposeState = (Q: false, K: false, V: false, O: false)

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("MFA Error: No Metal device available")
        return nil
    }

    do {
        let cached = try CachedAttentionKernel(
            descriptor: desc,
            device: device,
            includeBackward: include_backward
        )
        kernelCache[key] = cached
        return Unmanaged.passRetained(cached).toOpaque()
    } catch {
        print("MFA Error: Failed to create kernel: \(error)")
        return nil
    }
}

/// Execute forward attention
@_cdecl("mfa_execute_attention_forward")
public func mfa_execute_attention_forward(
    kernel_handle: MFAKernelHandle,
    command_buffer: MTLCommandBuffer,
    q_buffer: MTLBuffer,
    k_buffer: MTLBuffer,
    v_buffer: MTLBuffer,
    o_buffer: MTLBuffer,
    l_buffer: MTLBuffer,
    q_offset: Int,
    k_offset: Int,
    v_offset: Int,
    o_offset: Int,
    l_offset: Int,
    scale: Float
) -> Bool {
    let cached = Unmanaged<CachedAttentionKernel>.fromOpaque(kernel_handle).takeUnretainedValue()

    guard let encoder = command_buffer.makeComputeCommandEncoder() else {
        print("MFA Error: Failed to create compute encoder")
        return false
    }

    encoder.setComputePipelineState(cached.forwardPipeline)
    encoder.setThreadgroupMemoryLength(
        Int(cached.forwardKernel.threadgroupMemoryAllocation), index: 0)

    // Bind buffers using AttentionOperand buffer bindings from AttentionOperand.swift:
    // Q=0, K=1, V=2, O=3, L=4, D=5, dO=6, dV=7, dK=8, dQ=9
    encoder.setBuffer(q_buffer, offset: q_offset, index: 0)  // Q
    encoder.setBuffer(k_buffer, offset: k_offset, index: 1)  // K
    encoder.setBuffer(v_buffer, offset: v_offset, index: 2)  // V
    encoder.setBuffer(o_buffer, offset: o_offset, index: 3)  // O
    encoder.setBuffer(l_buffer, offset: l_offset, index: 4)  // L

    // Dispatch
    let seqLen = Int(cached.descriptor.matrixDimensions!.row)
    let blockCount = (seqLen + Int(cached.forwardKernel.blockDimensions.parallelization) - 1)
        / Int(cached.forwardKernel.blockDimensions.parallelization)

    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
    let groupSize = MTLSize(width: Int(cached.forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()

    return true
}

/// Execute backward attention (computes dQ, dK, dV)
@_cdecl("mfa_execute_attention_backward")
public func mfa_execute_attention_backward(
    kernel_handle: MFAKernelHandle,
    command_buffer: MTLCommandBuffer,
    q_buffer: MTLBuffer,
    k_buffer: MTLBuffer,
    v_buffer: MTLBuffer,
    o_buffer: MTLBuffer,
    l_buffer: MTLBuffer,
    do_buffer: MTLBuffer,
    dq_buffer: MTLBuffer,
    dk_buffer: MTLBuffer,
    dv_buffer: MTLBuffer,
    offset: Int,
    scale: Float
) -> Bool {
    let cached = Unmanaged<CachedAttentionKernel>.fromOpaque(kernel_handle).takeUnretainedValue()

    guard let bwdQueryPipeline = cached.backwardQueryPipeline,
          let bwdKVPipeline = cached.backwardKVPipeline,
          let bwdQueryKernel = cached.backwardQueryKernel,
          let bwdKVKernel = cached.backwardKVKernel else {
        print("MFA Error: Backward kernels not available")
        return false
    }

    let seqLen = Int(cached.descriptor.matrixDimensions!.row)

    // Execute backward query kernel (computes dQ)
    if let encoder = command_buffer.makeComputeCommandEncoder() {
        encoder.setComputePipelineState(bwdQueryPipeline)
        encoder.setThreadgroupMemoryLength(
            Int(bwdQueryKernel.threadgroupMemoryAllocation), index: 0)

        // Bind all required buffers (using AttentionOperand buffer bindings)
        // Q=0, K=1, V=2, O=3, L=4, D=5, dO=6, dV=7, dK=8, dQ=9
        encoder.setBuffer(q_buffer, offset: offset, index: 0)   // Q
        encoder.setBuffer(k_buffer, offset: offset, index: 1)   // K
        encoder.setBuffer(v_buffer, offset: offset, index: 2)   // V
        encoder.setBuffer(o_buffer, offset: offset, index: 3)   // O
        encoder.setBuffer(l_buffer, offset: offset, index: 4)   // L
        encoder.setBuffer(do_buffer, offset: offset, index: 6)  // dO
        encoder.setBuffer(dq_buffer, offset: offset, index: 9)  // dQ

        var scaleValue = scale
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.size, index: 10)

        let blockCount = (seqLen + Int(bwdQueryKernel.blockDimensions.parallelization) - 1)
            / Int(bwdQueryKernel.blockDimensions.parallelization)
        let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
        let groupSize = MTLSize(width: Int(bwdQueryKernel.threadgroupSize), height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()
    }

    // Execute backward KV kernel (computes dK, dV)
    if let encoder = command_buffer.makeComputeCommandEncoder() {
        encoder.setComputePipelineState(bwdKVPipeline)
        encoder.setThreadgroupMemoryLength(
            Int(bwdKVKernel.threadgroupMemoryAllocation), index: 0)

        // Q=0, K=1, V=2, O=3, L=4, D=5, dO=6, dV=7, dK=8, dQ=9
        encoder.setBuffer(q_buffer, offset: offset, index: 0)   // Q
        encoder.setBuffer(k_buffer, offset: offset, index: 1)   // K
        encoder.setBuffer(v_buffer, offset: offset, index: 2)   // V
        encoder.setBuffer(o_buffer, offset: offset, index: 3)   // O
        encoder.setBuffer(l_buffer, offset: offset, index: 4)   // L
        encoder.setBuffer(do_buffer, offset: offset, index: 6)  // dO
        encoder.setBuffer(dv_buffer, offset: offset, index: 7)  // dV
        encoder.setBuffer(dk_buffer, offset: offset, index: 8)  // dK

        var scaleValue = scale
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.size, index: 10)

        let blockCount = (seqLen + Int(bwdKVKernel.blockDimensions.parallelization) - 1)
            / Int(bwdKVKernel.blockDimensions.parallelization)
        let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
        let groupSize = MTLSize(width: Int(bwdKVKernel.threadgroupSize), height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()
    }

    return true
}

/// Release kernel handle (decrements reference count)
@_cdecl("mfa_release_kernel")
public func mfa_release_kernel(kernel_handle: MFAKernelHandle) {
    Unmanaged<CachedAttentionKernel>.fromOpaque(kernel_handle).release()
}

/// Clear kernel cache
@_cdecl("mfa_clear_cache")
public func mfa_clear_cache() {
    cacheLock.lock()
    defer { cacheLock.unlock() }
    kernelCache.removeAll()
}
