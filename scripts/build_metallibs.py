#!/usr/bin/env python3
"""
Build-time script to pre-compile all metallib files.

Run this during package build/install to ship pre-compiled kernels.
The metallibs are stored in mps_flash_attn/kernels/ and loaded directly at runtime.
"""

import os
import sys
import subprocess
import hashlib
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def build_metallibs():
    """Pre-compile Metal kernels to metallib files."""

    # Output directory (shipped with package)
    output_dir = Path(__file__).parent.parent / "mps_flash_attn" / "kernels"
    output_dir.mkdir(exist_ok=True)

    # We need to call Swift to generate the shader source
    swift_script = '''
import Foundation
import FlashAttention

struct Config: Codable {
    let seqLen: Int
    let headDim: Int
    let lowPrecision: Bool
    let source: String
    let sourceHash: String
}

var configs: [Config] = []

let seqLens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
let headDims = [32, 48, 64, 80, 96, 128]

for seqLen in seqLens {
    for headDim in headDims {
        for lowPrecision in [false, true] {
            var desc = AttentionDescriptor()
            desc.lowPrecisionInputs = lowPrecision
            desc.lowPrecisionIntermediates = lowPrecision
            desc.lowPrecisionOutputs = lowPrecision
            desc.matrixDimensions = (
                row: UInt32(seqLen),
                column: UInt32(seqLen),
                head: UInt16(headDim)
            )
            desc.transposeState = (Q: false, K: false, V: false, O: false)

            let kernelDesc = desc.kernelDescriptor(type: .forward)
            let kernel = AttentionKernel(descriptor: kernelDesc)
            let source = kernel.createSource()

            // Hash the source
            let data = source.data(using: .utf8)!
            var hash = [UInt8](repeating: 0, count: 32)
            _ = data.withUnsafeBytes { ptr in
                CC_SHA256(ptr.baseAddress, CC_LONG(data.count), &hash)
            }
            let hashString = hash.map { String(format: "%02x", $0) }.joined()

            configs.append(Config(
                seqLen: seqLen,
                headDim: headDim,
                lowPrecision: lowPrecision,
                source: source,
                sourceHash: hashString
            ))
        }
    }
}

let encoder = JSONEncoder()
encoder.outputFormatting = .prettyPrinted
let data = try! encoder.encode(configs)
print(String(data: data, encoding: .utf8)!)
'''

    print("Generating kernel configurations...")
    print("This will compile Metal shaders to .metallib files")
    print(f"Output directory: {output_dir}")
    print()

    # For now, use the existing cache mechanism but copy to package dir
    bridge_path = Path(__file__).parent.parent / "swift-bridge" / ".build" / "release" / "libMFABridge.dylib"

    if not bridge_path.exists():
        print("Building Swift bridge first...")
        subprocess.run(
            ["swift", "build", "-c", "release"],
            cwd=Path(__file__).parent.parent / "swift-bridge",
            check=True
        )

    import ctypes
    lib = ctypes.CDLL(str(bridge_path))

    print("Running pre-compilation via Swift bridge...")
    lib.mfa_init()
    lib.mfa_precompile()

    # Copy cached metallibs to package directory
    cache_dir = Path.home() / "Library" / "Application Support" / "MFABridge" / "metallib_cache"
    pipeline_dir = Path.home() / "Library" / "Application Support" / "MFABridge" / "pipeline_cache"

    if cache_dir.exists():
        import shutil
        for f in cache_dir.glob("*.metallib"):
            dest = output_dir / f.name
            shutil.copy2(f, dest)
            print(f"  Copied {f.name}")

    if pipeline_dir.exists():
        for f in pipeline_dir.glob("*.bin"):
            dest = output_dir / f.name
            shutil.copy2(f, dest)
            print(f"  Copied {f.name}")

    # Create manifest
    manifest = {
        "version": "1.0",
        "files": [f.name for f in output_dir.iterdir() if f.suffix in (".metallib", ".bin")]
    }

    import json
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest['files'])} files written to {output_dir}")
    print("These will be shipped with the package for zero-compilation runtime.")

if __name__ == "__main__":
    build_metallibs()
