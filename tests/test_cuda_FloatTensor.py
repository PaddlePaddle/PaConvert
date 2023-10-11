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

obj = APIBase("torch.cuda.FloatTensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if not torch.cuda.is_available():
            result = 1
        else:
            result = torch.cuda.FloatTensor(2, 3)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        shape = [2, 3]
        if not torch.cuda.is_available():
            result = 1
        else:
            result = torch.cuda.FloatTensor(*shape)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim1, dim2 = 2, 3
        if not torch.cuda.is_available():
            result = 1
        else:
            result = torch.cuda.FloatTensor(dim1, dim2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if not torch.cuda.is_available():
            result = 1
        else:
            result = torch.cuda.FloatTensor([[3, 4], [5, 8]])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if not torch.cuda.is_available():
            result = 1
        else:
            result = torch.cuda.FloatTensor((1, 2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if not torch.cuda.is_available():
            result = 1
        else:
            result = torch.cuda.FloatTensor()
        """
    )
    obj.run(pytorch_code, ["result"])
