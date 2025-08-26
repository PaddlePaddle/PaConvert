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

obj = APIBase("torch.Tensor.bmm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
        b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
        result = a.bmm(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]]).bmm(torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
        b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
        result = a.bmm(mat2=b)
        """
    )
    obj.run(pytorch_code, ["result"])


# The paddle input does not support integer type
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[4, 5, 6], [1, 2, 3]]])
        b = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        result = a.bmm(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]],
                          [[17., 18., 19., 20.], [21., 22., 23., 24.], [25., 26., 27., 28.], [29., 30., 31., 32.]],
                          [[33., 34., 35., 36.], [37., 38., 39., 40.], [41., 42., 43., 44.], [45., 46., 47., 48.]]
        ])
        b = torch.tensor([[[4., 5., 6.], [2., 3., 4.], [3., 3., 3.], [2., 2., 2.]],
                          [[8., 10., 11.], [5., 6., 8.], [4., 4., 4.], [1., 1., 1.]],
                          [[12., 13., 15.], [9., 10., 11.], [6., 6., 6.], [3., 3., 3.]]
        ])
        result = a.bmm(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]],
                          [[17., 18., 19., 20.], [21., 22., 23., 24.], [25., 26., 27., 28.], [29., 30., 31., 32.]],
                          [[33., 34., 35., 36.], [37., 38., 39., 40.], [41., 42., 43., 44.], [45., 46., 47., 48.]]
        ])
        b = torch.tensor([[[4., 5., 6.], [2., 3., 4.], [3., 3., 3.], [2., 2., 2.]],
                          [[8., 10., 11.], [5., 6., 8.], [4., 4., 4.], [1., 1., 1.]],
                          [[12., 13., 15.], [9., 10., 11.], [6., 6., 6.], [3., 3., 3.]]
        ])
        result = a.bmm(mat2=b)
        """
    )
    obj.run(pytorch_code, ["result"])
