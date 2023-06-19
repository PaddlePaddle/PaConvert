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

obj = APIBase("torch.nn.functional.max_unpool1d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[0.58987975, 0.80133516, 0.71605772, 0.46068805, 0.30434567, 0.41771618, 0.15606387, 0.88071585],
                        [0.67178625, 0.54522562, 0.83222342, 0.26114768, 0.77833325, 0.52892995, 0.26498035, 0.97040081]]])
        indices = torch.tensor([[[1 , 3 , 4 , 7 , 8 , 10, 13, 14],
                                [1 , 2 , 5 , 6 , 8 , 11, 13, 14]]])
        result = F.max_unpool1d(x, indices, kernel_size=2, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[0.58987975, 0.80133516, 0.71605772, 0.46068805, 0.30434567, 0.41771618, 0.15606387, 0.88071585],
                        [0.67178625, 0.54522562, 0.83222342, 0.26114768, 0.77833325, 0.52892995, 0.26498035, 0.97040081]]])
        indices = torch.tensor([[[1 , 3 , 4 , 7 , 8 , 10, 13, 14],
                                [1 , 2 , 5 , 6 , 8 , 11, 13, 14]]])
        result = F.max_unpool1d(x, indices, kernel_size=2, stride=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[0.58987975, 0.80133516, 0.71605772, 0.46068805, 0.30434567, 0.41771618, 0.15606387, 0.88071585],
                        [0.67178625, 0.54522562, 0.83222342, 0.26114768, 0.77833325, 0.52892995, 0.26498035, 0.97040081]]])
        indices = torch.tensor([[[1 , 3 , 4 , 7 , 8 , 10, 13, 14],
                                [1 , 2 , 5 , 6 , 8 , 11, 13, 14]]])
        result = F.max_unpool1d(x, indices, kernel_size=2, stride=2, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[0.5023, 0.7704]]])
        indices = torch.tensor([[[0, 3]]])
        result = F.max_unpool1d(x, indices, kernel_size=2, output_size=(1, 1, 4))
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle will generate error when the output_size parameter is specified",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[0.58987975, 0.80133516, 0.71605772, 0.46068805, 0.30434567, 0.41771618, 0.15606387, 0.88071585],
                        [0.67178625, 0.54522562, 0.83222342, 0.26114768, 0.77833325, 0.52892995, 0.26498035, 0.97040081]]])
        indices = torch.tensor([[[1 , 3 , 4 , 7 , 8 , 10, 13, 14],
                                [1 , 2 , 5 , 6 , 8 , 11, 13, 14]]])
        result = F.max_unpool1d(x, indices, kernel_size=2, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])
