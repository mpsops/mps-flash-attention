"""
Setup script for MPS Flash Attention
"""

import os
import sys
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class ObjCppBuildExt(build_ext):
    """Build extension for Objective-C++ with PyTorch."""
    def build_extensions(self):
        # Import torch only when actually building
        import torch
        from torch.utils import cpp_extension

        # Register .mm as a valid source extension
        self.compiler.src_extensions.append('.mm')

        # Get original compile function
        original_compile = self.compiler._compile

        def objcpp_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.mm'):
                # Force Objective-C++ mode for .mm files
                extra_postargs = ['-x', 'objective-c++'] + list(extra_postargs or [])
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = objcpp_compile

        for ext in self.extensions:
            ext.include_dirs.extend(cpp_extension.include_paths())
            ext.library_dirs.extend(cpp_extension.library_paths())
            ext.libraries.extend(['c10', 'torch', 'torch_cpu', 'torch_python'])

        super().build_extensions()

        # Copy libMFABridge.dylib to lib/ after building
        self._copy_swift_bridge()

    def _copy_swift_bridge(self):
        """Copy Swift bridge dylib to package lib/ directory."""
        src_path = os.path.join(
            os.path.dirname(__file__),
            "swift-bridge", ".build", "release", "libMFABridge.dylib"
        )
        dst_dir = os.path.join(os.path.dirname(__file__), "mps_flash_attn", "lib")
        dst_path = os.path.join(dst_dir, "libMFABridge.dylib")

        if os.path.exists(src_path):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"Copied libMFABridge.dylib to {dst_path}")
        else:
            print(f"Warning: {src_path} not found. Build swift-bridge first with:")
            print("  cd swift-bridge && swift build -c release")


def get_extensions():
    if sys.platform != "darwin":
        return []

    return [Extension(
        name="mps_flash_attn._C",
        sources=["mps_flash_attn/csrc/mps_flash_attn.mm"],
        extra_compile_args=["-std=c++17", "-O3", "-DTORCH_EXTENSION_NAME=_C"],
        extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
    )]


setup(
    name="mps-flash-attn",
    version="0.1.5",
    packages=find_packages(),
    package_data={
        "mps_flash_attn": [
            "lib/*.dylib",
            "kernels/*.metallib",
            "kernels/*.bin",
            "kernels/*.json",
        ],
    },
    include_package_data=True,
    install_requires=["torch>=2.0"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": ObjCppBuildExt},
)
