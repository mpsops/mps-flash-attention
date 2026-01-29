#!/usr/bin/env swift
/**
 * Standalone test for metal-flash-attention
 *
 * Run with: swift test_mfa_standalone.swift
 * Or: swiftc -O test_mfa_standalone.swift -o test_mfa && ./test_mfa
 */

import Foundation
import Metal

// We need to import the FlashAttention module from metal-flash-attention
// Since we can't easily do that from a standalone script, let's verify Metal works first

print("=== Metal Flash Attention Standalone Test ===\n")

// Check Metal device
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device found!")
}
print("✓ Metal device: \(device.name)")
print("  - Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
print("  - Max buffer length: \(device.maxBufferLength / 1024 / 1024) MB")

// Check if we're on Apple Silicon
#if arch(arm64)
print("✓ Running on Apple Silicon (arm64)")
#else
print("⚠ Not running on Apple Silicon")
#endif

// Check GPU family for Flash Attention support
if device.supportsFamily(.apple7) {
    print("✓ Supports Apple7 GPU family (M1+)")
} else {
    print("⚠ Does not support Apple7 GPU family")
}

if device.supportsFamily(.apple8) {
    print("✓ Supports Apple8 GPU family (M2+)")
}

if device.supportsFamily(.apple9) {
    print("✓ Supports Apple9 GPU family (M3+)")
}

// Test basic Metal compute
print("\n--- Testing basic Metal compute ---")

let commandQueue = device.makeCommandQueue()!

// Simple shader to verify Metal compute works
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void simple_add(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
"""

do {
    let library = try device.makeLibrary(source: shaderSource, options: nil)
    let function = library.makeFunction(name: "simple_add")!
    let pipeline = try device.makeComputePipelineState(function: function)

    let count = 1024
    let bufferA = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
    let bufferB = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
    let bufferC = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!

    // Fill input buffers
    let ptrA = bufferA.contents().bindMemory(to: Float.self, capacity: count)
    let ptrB = bufferB.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count {
        ptrA[i] = Float(i)
        ptrB[i] = Float(i * 2)
    }

    // Execute
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)
    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Verify result
    let ptrC = bufferC.contents().bindMemory(to: Float.self, capacity: count)
    var correct = true
    for i in 0..<count {
        if ptrC[i] != Float(i) + Float(i * 2) {
            correct = false
            break
        }
    }

    if correct {
        print("✓ Metal compute works correctly!")
    } else {
        print("✗ Metal compute produced incorrect results")
    }
} catch {
    print("✗ Failed to create Metal pipeline: \(error)")
}

print("\n=== Next Steps ===")
print("To use metal-flash-attention, we need to:")
print("1. Build it as a Swift Package")
print("2. Link against the FlashAttention module")
print("3. Call AttentionDescriptor -> AttentionKernel -> createSource()")
print("")
print("Run: cd metal-flash-attention && swift build")
