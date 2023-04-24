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

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.dtype')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 2, 3], dtype=torch.float16)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.bool)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.half)
        '''
    )
    obj.run(pytorch_code, ["result"])

# numpy not supports bfloat, but paddle and pytorch support and they are equal
def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.bfloat16)
        '''
    )
    obj.run(pytorch_code, ["result"])

# torch has no attribute uint16
def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.uint16)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.float32)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_7():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.float)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_8():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.float64)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_9():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.double)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_10():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.int8)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_11():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.int16)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_12():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.short)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_13():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.int32)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_14():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.int)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_15():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.int64)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_16():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.long)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_17():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.uint8)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_18():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.complex64)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_19():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.cfloat)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_20():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.complex128)
        '''
    )
    obj.run(pytorch_code, ["result"])

def test_case_21():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1, 0, 3], dtype=torch.cdouble)
        '''
    )
    obj.run(pytorch_code, ["result"])
