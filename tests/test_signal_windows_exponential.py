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

obj = APIBase("torch.signal.windows.exponential")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = 10
        result = torch.signal.windows.exponential(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True, dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True, dtype=torch.float64, layout=torch.strided)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, sym=True, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(M=10, tau=0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(10, tau=0.5, sym=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(10, sym=False, tau=0.5, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(10, tau=0.5, sym=False, dtype=torch.float64, layout=torch.strided, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(10, tau=0.5, sym=False, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(10, tau=0.5, sym=False, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(requires_grad=False, sym=True, M=10, layout=torch.strided, device=torch.device('cpu'), tau=0.6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.signal.windows.exponential(requires_grad=True, sym=False, M=10, layout=torch.strided, device=torch.device('cpu'), tau=0.3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=False
        result = torch.signal.windows.exponential(requires_grad=True, sym=a, M=10, layout=torch.strided, device=torch.device('cpu'), tau=0.3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=False
        m=10
        result = torch.signal.windows.exponential(requires_grad=True, sym=a, M=m, layout=torch.strided, device=torch.device('cpu'), tau=0.3)
        """
    )
    obj.run(pytorch_code, ["result"])
