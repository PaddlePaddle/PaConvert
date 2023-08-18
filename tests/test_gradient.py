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

obj = APIBase("torch.gradient")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        values = torch.tensor([4., 1., 1., 16.], )
        coordinates = (torch.tensor([-2., -1., 1., 4.]),)
        result = torch.gradient(values, spacing = coordinates)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]])
        result = torch.gradient(t)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 4, 8], [10, 20, 40, 80]])
        result = torch.gradient(t, spacing = 2.0)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
