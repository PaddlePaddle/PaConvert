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

obj = APIBase("torch.where")


# when y is a float scalar, paddle.where will return float64
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = 10.
        result = torch.where(x>0, x, y)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


# paddle.where not support type promotion between x and y, while torch.where support
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = 10
        result = torch.where(x>0, x, y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = torch.tensor(10.)
        result = torch.where(x>0, x, y)
        """
    )
    obj.run(pytorch_code, ["result"])


# torch.where(x) means torch.nonzero(x, as_tuple=True)
# but torch.nonzero(x) return different shape with paddle.nonzero(x)
# paddle.nonzero(x) return [N, 1], shoule be fixed to [N]
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[3, 0], [4.8, 9.2]])
        result = torch.where(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_shape=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        result = torch.where(condition=x>0, input=x, other=y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = torch.zeros((1, 4))
        result = torch.where(condition=x>0, input=x, other=y, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = torch.zeros((1, 4))
        result = torch.where(input=x, other=y, condition=x>0, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
