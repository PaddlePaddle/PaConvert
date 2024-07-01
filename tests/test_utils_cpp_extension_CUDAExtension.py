# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import textwrap

import pytest
import torch
from apibase import APIBase

obj = APIBase("torch.utils.cpp_extension.CUDAExtension")


@pytest.mark.skipif(
    condition=not torch.backends.cuda.is_built(),
    reason="can only run on torch with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import CUDAExtension

        CUDAExtension(
                name='cuda_extension',
                sources=['extension.cpp', 'extension_kernel.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']})
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not torch.backends.cuda.is_built(),
    reason="can only run on torch with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import CUDAExtension

        CUDAExtension(
                'cuda_extension',
                ['extension.cpp', 'extension_kernel.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']})
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])
