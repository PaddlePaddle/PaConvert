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

obj = APIBase("torch.nn.MaxPool1d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
        model = nn.MaxPool1d(2)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
        model = nn.MaxPool1d(2, 1)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
        model = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
        model = nn.MaxPool1d(kernel_size=2, stride=1, padding=1, ceil_mode=True)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[0.1, 1., 2., 3.], [4., 5., 6., 7.]]])
        model = nn.MaxPool1d(kernel_size=2, stride=1, padding=1, return_indices=True)
        result, indices = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle.nn.MaxPool1D dose not support 'dilation' now!
def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
        model = nn.MaxPool1d(kernel_size=2, stride=1, dialation=2)
        result, indices = model(x)
        """
    )
    obj.run(pytorch_code, ["result", "indices"])
