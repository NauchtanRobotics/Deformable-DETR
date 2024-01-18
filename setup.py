# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup
from pathlib import Path


requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = Path(__file__).parent
    extensions_dir = this_dir / "deformable_detr" / "models" / "ops" / "src"

    main_file = extensions_dir.glob("*.cpp")
    source_cpu_dir = extensions_dir / "cpu"
    source_cpu = source_cpu_dir.glob("*.cpp")

    source_cuda_dir = extensions_dir / "cuda"
    source_cuda = source_cuda_dir.glob("*.cu")

    sources = list(main_file) + list(source_cpu) + list(source_cuda)
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is None:
        raise NotImplementedError("CUDA_HOME not resolved")

    if not torch.cuda.is_available():
        raise NotImplementedError('Cuda is not available')

    extension = CUDAExtension
    sources = [str(s) for s in sources]
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    sources = [str(s) for s in sources]
    include_dirs = [str(extensions_dir)]
    print("sources: ")
    print(sources)
    print("include_dirs: ")
    print(include_dirs)
    print()
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
