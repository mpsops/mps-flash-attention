/**
 * Test FlashAttention from metal-flash-attention
 * Build and run:
 *   swiftc -O -I .build/release -L .build/release -lFlashAttention test_flash_attention.swift -o test_fa && ./test_fa
 */

import Foundation
import Metal
import FlashAttention

print("=== Flash Attention Test ===\n")

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}
print("Device: \(device.name)")

let commandQueue = device.makeCommandQueue()!

// Test parameters
let seqLen = 128
let headDim = 64
let dtype = "float32"

print("\nTest config: seqLen=\(seqLen), headDim=\(headDim), dtype=\(dtype)")

// Create attention descriptor
var attentionDesc = AttentionDescriptor()
attentionDesc.lowPrecisionInputs = false  // FP32
attentionDesc.lowPrecisionIntermediates = false
attentionDesc.matrixDimensions = (
    row: UInt32(seqLen),
    column: UInt32(seqLen),
    head: UInt16(headDim)
)
attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)

print("\n--- Creating Forward Kernel ---")
let forwardKernelDesc = attentionDesc.kernelDescriptor(type: .forward)
let forwardKernel = AttentionKernel(descriptor: forwardKernelDesc)

print("Block dimensions: parallelization=\(forwardKernel.blockDimensions.parallelization), traversal=\(forwardKernel.blockDimensions.traversal), head=\(forwardKernel.blockDimensions.head)")
print("Threadgroup size: \(forwardKernel.threadgroupSize)")
print("Threadgroup memory: \(forwardKernel.threadgroupMemoryAllocation) bytes")

// Generate and compile Metal shader
print("\n--- Compiling Metal Shader ---")
let shaderSource = forwardKernel.createSource()

// Print first 500 chars of shader for debugging
print("Shader preview:")
print(String(shaderSource.prefix(500)))
print("...")

do {
    let library = try device.makeLibrary(source: shaderSource, options: nil)

    // Set function constants
    let constants = MTLFunctionConstantValues()
    attentionDesc.setFunctionConstants(constants)

    let function = try library.makeFunction(name: "attention", constantValues: constants)

    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024

    let pipeline = try device.makeComputePipelineState(descriptor: pipelineDesc, options: [], reflection: nil)

    print("✓ Shader compiled successfully!")
    print("  Max total threads: \(pipeline.maxTotalThreadsPerThreadgroup)")

    // Allocate buffers
    let qkvSize = seqLen * headDim * MemoryLayout<Float>.size
    let lSize = seqLen * MemoryLayout<Float>.size

    let bufferQ = device.makeBuffer(length: qkvSize, options: .storageModeShared)!
    let bufferK = device.makeBuffer(length: qkvSize, options: .storageModeShared)!
    let bufferV = device.makeBuffer(length: qkvSize, options: .storageModeShared)!
    let bufferO = device.makeBuffer(length: qkvSize, options: .storageModeShared)!
    let bufferL = device.makeBuffer(length: lSize, options: .storageModeShared)!

    // Fill Q, K, V with random data
    func fillRandom(_ buffer: MTLBuffer, count: Int) {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float.random(in: -1...1)
        }
    }
    fillRandom(bufferQ, count: seqLen * headDim)
    fillRandom(bufferK, count: seqLen * headDim)
    fillRandom(bufferV, count: seqLen * headDim)

    print("\n--- Running Forward Pass ---")
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(Int(forwardKernel.threadgroupMemoryAllocation), index: 0)

    // Bind buffers: Q=0, K=1, V=2, O=3, L=4
    encoder.setBuffer(bufferQ, offset: 0, index: 0)
    encoder.setBuffer(bufferK, offset: 0, index: 1)
    encoder.setBuffer(bufferV, offset: 0, index: 2)
    encoder.setBuffer(bufferO, offset: 0, index: 3)
    encoder.setBuffer(bufferL, offset: 0, index: 4)

    // Dispatch
    let blockCount = (seqLen + Int(forwardKernel.blockDimensions.parallelization) - 1)
        / Int(forwardKernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
    let groupSize = MTLSize(width: Int(forwardKernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()

    let start = CFAbsoluteTimeGetCurrent()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if let error = commandBuffer.error {
        print("✗ GPU error: \(error)")
    } else {
        print("✓ Forward pass completed in \(String(format: "%.2f", elapsed * 1000)) ms")

        // Check output is not all zeros/NaN
        let ptrO = bufferO.contents().bindMemory(to: Float.self, capacity: seqLen * headDim)
        var hasNaN = false
        var allZero = true
        var sum: Float = 0
        for i in 0..<(seqLen * headDim) {
            if ptrO[i].isNaN { hasNaN = true }
            if ptrO[i] != 0 { allZero = false }
            sum += abs(ptrO[i])
        }

        if hasNaN {
            print("⚠ Output contains NaN values")
        } else if allZero {
            print("⚠ Output is all zeros")
        } else {
            print("✓ Output looks valid (mean abs: \(sum / Float(seqLen * headDim)))")
        }

        // Check logsumexp
        let ptrL = bufferL.contents().bindMemory(to: Float.self, capacity: seqLen)
        var lSum: Float = 0
        for i in 0..<seqLen {
            lSum += ptrL[i]
        }
        print("  LogSumExp mean: \(lSum / Float(seqLen))")
    }

    print("\n=== Flash Attention works! ===")

} catch {
    print("✗ Failed: \(error)")
}
