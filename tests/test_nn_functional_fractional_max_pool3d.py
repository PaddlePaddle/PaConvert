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

obj = APIBase("torch.nn.functional.fractional_max_pool3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[[ 1.1524,  0.4714,  0.2857],
                    [-1.2533, -0.9829, -1.0981],
                    [ 0.1507, -1.1431, -2.0361]],

                [[ 0.1024, -0.4482,  0.4137],
                    [ 0.9385,  0.4565,  0.7702],
                    [ 0.4135, -0.2587,  0.0482]]]]])
        random_samples = torch.tensor([[[ 0.7,  0.3],[ 0.7,  0.3]],[[ 0.7,  0.3],[ 0.7,  0.3]]])
        result = torch.nn.functional.fractional_max_pool3d(input, output_ratio=(0.7,0.7,0.7), kernel_size=1, return_indices=False, _random_samples=random_samples)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support `_random_samples`",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.rand(1, 1, 6, 6, 6)
        result = torch.nn.functional.fractional_max_pool3d(input, output_ratio=(0.7,0.7,0.7), kernel_size=1, return_indices=False)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.rand(1, 1, 6, 6, 6)
        result = torch.nn.functional.fractional_max_pool3d(input, output_size=(1,2,2), kernel_size=1, return_indices=False)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.rand(5, 16, 15, 20, 30)
        result = torch.nn.functional.fractional_max_pool3d(input=input, kernel_size=1, output_size=(5,5,5))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.rand(5, 16, 15, 20, 30)
        result = torch.nn.functional.fractional_max_pool3d(input, 1, (5,5,5))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.rand(1, 1, 6, 6, 6)
        result = torch.nn.functional.fractional_max_pool3d(input, 1, None, (0.7,0.7,0.7), False)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.rand(1, 1, 6, 6, 6)
        result, indices = torch.nn.functional.fractional_max_pool3d(input, output_size=(1,2,2), kernel_size=1, return_indices=True)
        """
    )
    obj.run(pytorch_code, ["result", "indices"], check_value=False)
