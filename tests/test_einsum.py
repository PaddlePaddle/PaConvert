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

obj = APIBase("torch.einsum")


def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3],[6, 2, 9], [1, 2, 3]])
        result = torch.einsum('ij->', x)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [6, 2, 9]])
        result = torch.einsum('ij->ji', x)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3],[6, 2, 9]])
        y = torch.tensor([[1, 2],[6, 2], [5, 9]])
        result = torch.einsum('ij,jk->ik', x, y)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1, 2, 3],[6, 2, 9]]])
        y = torch.tensor([[[1, 2],[6, 2], [5, 9]]])
        result = torch.einsum('...ij,...jk->...ik', x, y)
        """
    )
    obj.run(pytorch_code, ["result"])
