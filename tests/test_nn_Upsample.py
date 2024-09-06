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

obj = APIBase("torch.nn.Upsample")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(scale_factor=2, mode='nearest')
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(size=(2,2))
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(scale_factor=2, mode='bilinear',recompute_scale_factor=True)
        result = m(input)
        """
    )
    obj.run(
        pytorch_code, unsupport=True, reason="paddle unsupport recompute_scale_factor "
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]]])
        m = torch.nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])
