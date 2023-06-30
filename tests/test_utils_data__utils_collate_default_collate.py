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

obj = APIBase("torch.utils.data._utils.collate.default_collate")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data._utils.collate import default_collate
        result = torch.tensor(default_collate([0, 1, 2, 3]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data._utils.collate import default_collate
        result = default_collate(['a', 'b', 'c'])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data._utils.collate import default_collate
        result = default_collate([torch.tensor([0, 1, 2, 3])])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data._utils.collate import default_collate
        result = default_collate((torch.tensor([1, 3, 3]), torch.tensor([3, 1, 1])))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data._utils.collate import default_collate
        result = default_collate(batch=(torch.tensor([1, 3, 3]), torch.tensor([3, 1, 1])))
        """
    )
    obj.run(pytorch_code, ["result"])


# Pytorch returns Tensor by default, while paddle returns narray.
def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.dataloader import default_collate
        result = default_collate(batch=default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])))
        """
    )
    obj.run(pytorch_code, ["result"])


# Pytorch returns Tensor by default, while paddle returns narray.
def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.dataloader import default_collate
        result = default_collate([0, 1, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])
