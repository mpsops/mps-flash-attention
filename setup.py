"""
Setup script for MPS Flash Attention
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Lazy import torch to avoid build isolation issues
def get_torch_extension():
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    return CppExtension, BuildExtension


def get_extensions():
    if sys.platform != "darwin":
        return []

    CppExtension, _ = get_torch_extension()

    this_dir = Path(__file__).parent
    csrc_dir = this_dir / "mps_flash_attn" / "csrc"

    sources = [str(csrc_dir / "mps_flash_attn.mm")]

    extra_compile_args = ["-std=c++17", "-O3"]
    extra_link_args = ["-framework", "Metal", "-framework", "Foundation"]

    return [CppExtension(
        name="mps_flash_attn._C",
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="objc++",
    )]


def get_cmdclass():
    _, BuildExtension = get_torch_extension()
    return {"build_ext": BuildExtension}


setup(
    name="mps-flash-attention",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch>=2.0"],
    ext_modules=get_extensions(),
    cmdclass=get_cmdclass(),
)
