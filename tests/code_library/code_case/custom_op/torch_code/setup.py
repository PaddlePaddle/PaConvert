import os
import glob

import torch

from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_sources(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(current_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    # source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = main_file
    sources += source_cuda
    sources = [os.path.join(extensions_dir, s) for s in sources]
    return extensions_dir, sources


def get_extensions():
    define_macros = []
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args = {"cxx": []}
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    ext_modules = [
        CUDAExtension(
            "GeometricKernelAttention",
            get_sources()[1],
            include_dirs=get_sources()[0],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="GeometricKernelAttention", version="1.0", author="Tianheng Cheng", url="https://github.com/hustvl",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Geometric Kernel Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
