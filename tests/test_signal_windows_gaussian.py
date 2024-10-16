# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.signal.windows.gaussian")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=10
        result = torch.signal.windows.gaussian(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=True, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=True, dtype=torch.float64, layout=torch.strided)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=True, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=True, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=True, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=False, layout=torch.strided, device=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, sym=False, layout=torch.strided, device=torch.device('cpu'), requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.gaussian(M=10, std=0.6, sym=False, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=True
        result = torch.signal.windows.gaussian(requires_grad=False, std=0.8, sym=a, M=10, layout=torch.strided, device=torch.device('cpu'), dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=False
        m=10
        result = torch.signal.windows.gaussian(requires_grad=False, std=0.8, sym=a, M=m, layout=torch.strided, device=torch.device('cpu'), dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-05)


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=False
        m=10
        s=0.8
        d=torch.float32
        e=torch.device('cpu')
        result = torch.signal.windows.gaussian(requires_grad=False, std=s, sym=a, M=m, layout=torch.strided, device=e, dtype=d)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-05)
