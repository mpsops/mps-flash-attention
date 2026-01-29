//
//  MetallibCache.swift
//  MFABridge
//
//  Pre-compiled metallib caching for zero compilation overhead.
//

import Foundation
import Metal
import FlashAttention
import CryptoKit

/// Manages pre-compiled metallib files and pipeline binaries for instant kernel loading
public class MetallibCache {

    /// Singleton instance
    public static let shared = MetallibCache()

    /// Directory where metallib files are stored
    private let cacheDir: URL

    /// Directory for binary archives (compiled pipelines)
    private let pipelineCacheDir: URL

    /// In-memory cache of loaded libraries
    private var libraryCache: [String: MTLLibrary] = [:]

    /// In-memory cache of pipeline states
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    private let lock = NSLock()

    /// Directory for pre-shipped kernels (read-only, in package)
    private var shippedKernelsDir: URL?

    private init() {
        // Use app support directory for runtime cache
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        cacheDir = appSupport.appendingPathComponent("MFABridge/metallib_cache", isDirectory: true)
        pipelineCacheDir = appSupport.appendingPathComponent("MFABridge/pipeline_cache", isDirectory: true)

        // Create cache directories if needed
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: pipelineCacheDir, withIntermediateDirectories: true)

        // Look for pre-shipped kernels directory
        // This would be set by the Python package during load
        if let envPath = ProcessInfo.processInfo.environment["MFA_KERNELS_DIR"] {
            shippedKernelsDir = URL(fileURLWithPath: envPath)
        }
    }

    /// Set the directory containing pre-shipped kernels
    public func setShippedKernelsDir(_ path: String) {
        shippedKernelsDir = URL(fileURLWithPath: path)
    }

    /// Get pipeline binary archive path
    private func pipelinePath(for key: String) -> URL {
        pipelineCacheDir.appendingPathComponent("\(key).bin")
    }

    /// Generate a unique key for a kernel configuration
    private func cacheKey(source: String) -> String {
        // SHA256 hash of the shader source
        let data = Data(source.utf8)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }

    /// Get metallib file path for a given source hash
    private func metallibPath(for key: String) -> URL {
        cacheDir.appendingPathComponent("\(key).metallib")
    }

    /// Load or compile a library, with disk caching
    public func getLibrary(source: String, device: MTLDevice) throws -> MTLLibrary {
        let key = cacheKey(source: source)

        lock.lock()
        defer { lock.unlock() }

        // Check in-memory cache first
        if let cached = libraryCache[key] {
            return cached
        }

        // Check disk cache
        let metallibURL = metallibPath(for: key)
        if FileManager.default.fileExists(atPath: metallibURL.path) {
            do {
                let library = try device.makeLibrary(URL: metallibURL)
                libraryCache[key] = library
                return library
            } catch {
                // Cached file corrupted, will recompile
                try? FileManager.default.removeItem(at: metallibURL)
            }
        }

        // Compile and cache
        let library = try compileAndCache(source: source, key: key, device: device)
        libraryCache[key] = library
        return library
    }

    /// Compile shader and save metallib to disk
    private func compileAndCache(source: String, key: String, device: MTLDevice) throws -> MTLLibrary {
        let fileManager = FileManager.default
        let tempDir = fileManager.temporaryDirectory
        let uuid = UUID().uuidString

        let sourceFile = tempDir.appendingPathComponent("mfa_\(uuid).metal")
        let airFile = tempDir.appendingPathComponent("mfa_\(uuid).air")
        let tempMetallib = tempDir.appendingPathComponent("mfa_\(uuid).metallib")
        let finalMetallib = metallibPath(for: key)

        defer {
            try? fileManager.removeItem(at: sourceFile)
            try? fileManager.removeItem(at: airFile)
            try? fileManager.removeItem(at: tempMetallib)
        }

        // Write source
        try source.write(to: sourceFile, atomically: true, encoding: .utf8)

        // Compile to AIR
        let compileResult = shell("xcrun metal -c '\(sourceFile.path)' -o '\(airFile.path)' 2>&1")
        guard compileResult.status == 0 else {
            throw NSError(domain: "MFABridge", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "metal compile failed: \(compileResult.output)"])
        }

        // Link to metallib
        let linkResult = shell("xcrun metallib '\(airFile.path)' -o '\(tempMetallib.path)' 2>&1")
        guard linkResult.status == 0 else {
            throw NSError(domain: "MFABridge", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "metallib link failed: \(linkResult.output)"])
        }

        // Move to cache location
        try? fileManager.removeItem(at: finalMetallib)
        try fileManager.copyItem(at: tempMetallib, to: finalMetallib)

        // Load and return
        return try device.makeLibrary(URL: finalMetallib)
    }

    private func shell(_ command: String) -> (output: String, status: Int32) {
        let task = Process()
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", command]

        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            return ("Failed: \(error)", -1)
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        return (output, task.terminationStatus)
    }

    /// Pre-compile common configurations (call at install/build time)
    public func precompile(device: MTLDevice, configurations: [(seqLen: Int, headDim: Int, lowPrecision: Bool)]) {
        print("Pre-compiling \(configurations.count) kernel configurations...")

        for (i, config) in configurations.enumerated() {
            do {
                var desc = AttentionDescriptor()
                desc.lowPrecisionInputs = config.lowPrecision
                desc.lowPrecisionIntermediates = config.lowPrecision
                desc.lowPrecisionOutputs = config.lowPrecision
                desc.matrixDimensions = (
                    row: UInt32(config.seqLen),
                    column: UInt32(config.seqLen),
                    head: UInt16(config.headDim)
                )
                desc.transposeState = (Q: false, K: false, V: false, O: false)

                let kernelDesc = desc.kernelDescriptor(type: .forward)
                let kernel = AttentionKernel(descriptor: kernelDesc)
                let source = kernel.createSource()

                _ = try getLibrary(source: source, device: device)
                print("  [\(i+1)/\(configurations.count)] seqLen=\(config.seqLen), headDim=\(config.headDim), fp16=\(config.lowPrecision) ✓")
            } catch {
                print("  [\(i+1)/\(configurations.count)] FAILED: \(error)")
            }
        }
        print("Pre-compilation complete!")
    }

    /// Generate pipeline cache key including constants
    private func pipelineCacheKey(sourceKey: String, rowDim: UInt32, colDim: UInt32) -> String {
        "\(sourceKey)_\(rowDim)_\(colDim)"
    }

    /// Get or create a compute pipeline with binary caching
    public func getPipeline(
        source: String,
        functionName: String,
        constants: MTLFunctionConstantValues,
        rowDim: UInt32,
        colDim: UInt32,
        device: MTLDevice
    ) throws -> MTLComputePipelineState {
        // Create unique key from source + constants
        let sourceKey = cacheKey(source: source)
        let key = pipelineCacheKey(sourceKey: sourceKey, rowDim: rowDim, colDim: colDim)

        lock.lock()
        defer { lock.unlock() }

        // Check in-memory cache
        if let cached = pipelineCache[key] {
            return cached
        }

        // Get or compile library (keyed by source only)
        let library: MTLLibrary
        if let cachedLib = libraryCache[sourceKey] {
            library = cachedLib
        } else {
            library = try getLibraryUnlocked(source: source, key: sourceKey, device: device)
            libraryCache[sourceKey] = library
        }

        // Try to load from binary archive (check shipped first, then cache)
        var archiveURL = pipelinePath(for: key)
        if let shipped = shippedKernelsDir {
            let shippedPipeline = shipped.appendingPathComponent("\(key).bin")
            if FileManager.default.fileExists(atPath: shippedPipeline.path) {
                archiveURL = shippedPipeline
            }
        }
        if FileManager.default.fileExists(atPath: archiveURL.path) {
            do {
                let archiveDesc = MTLBinaryArchiveDescriptor()
                archiveDesc.url = archiveURL
                let archive = try device.makeBinaryArchive(descriptor: archiveDesc)

                let pipelineDesc = MTLComputePipelineDescriptor()
                let function = try library.makeFunction(name: functionName, constantValues: constants)
                pipelineDesc.computeFunction = function
                pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
                pipelineDesc.binaryArchives = [archive]

                let pipeline = try device.makeComputePipelineState(
                    descriptor: pipelineDesc,
                    options: .failOnBinaryArchiveMiss,
                    reflection: nil
                )
                pipelineCache[key] = pipeline
                return pipeline
            } catch {
                // Archive miss or corrupted, fall through to compile
                try? FileManager.default.removeItem(at: archiveURL)
            }
        }

        // Compile pipeline
        let function = try library.makeFunction(name: functionName, constantValues: constants)
        let pipelineDesc = MTLComputePipelineDescriptor()
        pipelineDesc.computeFunction = function
        pipelineDesc.maxTotalThreadsPerThreadgroup = 1024

        let pipeline = try device.makeComputePipelineState(
            descriptor: pipelineDesc, options: [], reflection: nil)

        // Save to binary archive for next time
        do {
            let archiveDesc = MTLBinaryArchiveDescriptor()
            let archive = try device.makeBinaryArchive(descriptor: archiveDesc)
            try archive.addComputePipelineFunctions(descriptor: pipelineDesc)
            try archive.serialize(to: archiveURL)
        } catch {
            // Cache save failed, not critical
        }

        pipelineCache[key] = pipeline
        return pipeline
    }

    /// Internal: get library without lock (caller must hold lock)
    private func getLibraryUnlocked(source: String, key: String, device: MTLDevice) throws -> MTLLibrary {
        // 1. Check shipped kernels first (fastest - no compilation ever needed)
        if let shipped = shippedKernelsDir {
            let shippedURL = shipped.appendingPathComponent("\(key).metallib")
            if FileManager.default.fileExists(atPath: shippedURL.path) {
                do {
                    return try device.makeLibrary(URL: shippedURL)
                } catch {
                    // Shipped file corrupted? Fall through
                }
            }
        }

        // 2. Check runtime cache
        let metallibURL = metallibPath(for: key)
        if FileManager.default.fileExists(atPath: metallibURL.path) {
            do {
                return try device.makeLibrary(URL: metallibURL)
            } catch {
                try? FileManager.default.removeItem(at: metallibURL)
            }
        }

        // 3. Compile and cache (fallback for unseen configurations)
        return try compileAndCache(source: source, key: key, device: device)
    }

    /// Preload ALL cached pipelines into memory at startup.
    /// This eliminates any loading latency on first kernel call.
    public func preloadAllCached(device: MTLDevice) {
        let fileManager = FileManager.default

        lock.lock()
        defer { lock.unlock() }

        // Load all metallibs into memory
        if let metallibFiles = try? fileManager.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil) {
            for metallibURL in metallibFiles where metallibURL.pathExtension == "metallib" {
                let key = metallibURL.deletingPathExtension().lastPathComponent
                if libraryCache[key] == nil {
                    if let library = try? device.makeLibrary(URL: metallibURL) {
                        libraryCache[key] = library
                    }
                }
            }
        }

        // Load all pipeline binaries into memory
        // The binary archives are loaded when getPipeline is called,
        // but loading metallibs eagerly helps a lot already
    }

    /// Preload specific configurations for zero-latency access.
    /// Call this with the configs you expect to use.
    public func preloadConfigurations(
        device: MTLDevice,
        configurations: [(seqLen: Int, headDim: Int, lowPrecision: Bool)]
    ) {
        for config in configurations {
            do {
                var desc = AttentionDescriptor()
                desc.lowPrecisionInputs = config.lowPrecision
                desc.lowPrecisionIntermediates = config.lowPrecision
                desc.lowPrecisionOutputs = config.lowPrecision
                desc.matrixDimensions = (
                    row: UInt32(config.seqLen),
                    column: UInt32(config.seqLen),
                    head: UInt16(config.headDim)
                )
                desc.transposeState = (Q: false, K: false, V: false, O: false)

                let kernelDesc = desc.kernelDescriptor(type: .forward)
                let kernel = AttentionKernel(descriptor: kernelDesc)
                let source = kernel.createSource()

                let constants = MTLFunctionConstantValues()
                desc.setFunctionConstants(constants)

                // This will load from cache or compile
                _ = try getPipeline(
                    source: source,
                    functionName: "attention",
                    constants: constants,
                    rowDim: UInt32(config.seqLen),
                    colDim: UInt32(config.seqLen),
                    device: device
                )
            } catch {
                // Ignore errors during preload
            }
        }
    }

    /// Clear the cache
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        libraryCache.removeAll()
        pipelineCache.removeAll()
        try? FileManager.default.removeItem(at: cacheDir)
        try? FileManager.default.removeItem(at: pipelineCacheDir)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: pipelineCacheDir, withIntermediateDirectories: true)
    }
}
