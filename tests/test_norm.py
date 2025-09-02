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

from apibase import APIBase

obj = APIBase("torch.norm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[-12., -11., -10., -9. ],
                            [-8. , -7. , -6. , -5. ],
                            [-4. , -3. , -2. , -1. ]],
                            [[ 0. ,  1. ,  2. ,  3. ],
                            [ 4. ,  5. ,  6. ,  7. ],
                            [ 8. ,  9. ,  10.,  11.]]])
        result = torch.norm(input, p='fro')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        result = torch.norm(input, p='nuc')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        result = torch.norm(input, p=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        result = torch.norm(input, p=2, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        result = torch.norm(input, p=2, dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        result = torch.norm(input, p=2, dim=1, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        out = torch.tensor([1.], dtype=torch.float64)
        result = torch.norm(input, 2, 1, True, out, torch.float64)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        out = torch.tensor([1.], dtype=torch.float64)
        result = torch.norm(input=input, p=2, dim=1, keepdim=True, out=out, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]])
        out = torch.tensor([1.], dtype=torch.float64)
        result = torch.norm(input=input, keepdim=True, dim=1, p=2, out=out, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[-12., -11., -10., -9. ],
                            [-8. , -7. , -6. , -5. ],
                            [-4. , -3. , -2. , -1. ]],
                            [[ 0. ,  1. ,  2. ,  3. ],
                            [ 4. ,  5. ,  6. ,  7. ],
                            [ 8. ,  9. ,  10.,  11.]]])
        result = torch.norm(input)
        """
    )
    obj.run(pytorch_code, ["result"])
