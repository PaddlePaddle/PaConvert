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
#
import textwrap

from apibase import APIBase

obj = APIBase("torch.index_put")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones([5, 3])
        t = torch.tensor([1.], dtype=torch.float)
        indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
        result = torch.index_put(x, indices, t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones([5, 3])
        t = torch.tensor([1.], dtype=torch.float)
        indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
        result = torch.index_put(x, indices, values=t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones([5, 3])
        t = torch.tensor([1.], dtype=torch.float)
        indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
        result = torch.index_put(x, indices, t, True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones([5, 3])
        t = torch.tensor([1.], dtype=torch.float)
        indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
        result = torch.index_put(input=x, indices=indices, values=t, accumulate=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones([5, 3])
        t = torch.tensor([1.], dtype=torch.float)
        indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
        result = torch.index_put(indices=indices, values=t, accumulate=False, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones([5, 3])
        t = torch.tensor([1.], dtype=torch.float)
        indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
        result = torch.index_put(accumulate=True, input=x, indices=indices, values=t)
        """
    )
    obj.run(pytorch_code, ["result"])
