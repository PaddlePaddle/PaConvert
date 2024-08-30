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

obj = APIBase("torch.utils.data.BatchSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import BatchSampler
        from torch.utils.data import SequentialSampler
        batch_sampler = BatchSampler(sampler=SequentialSampler([3, 9, 10, 5, 7, 6, 1]), batch_size=3, drop_last=True)
        result = list(batch_sampler)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import BatchSampler
        batch_sampler = BatchSampler([3, 9, 10, 5, 7, 6, 1], batch_size=3, drop_last=False)
        result = list(batch_sampler)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        batch_sampler = torch.utils.data.BatchSampler([3, 9, 10, 5, 7, 6, 1], 2, True)
        result = list(batch_sampler)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        batch_size = 4
        batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler([3, 9, 10, 5, 7, 6, 1]), batch_size, False)
        result = list(batch_sampler)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = list(torch.utils.data.BatchSampler(batch_size=4, drop_last=False, sampler=[3, 9, 10, 5, 7, 6, 1]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import BatchSampler
        from torch.utils.data import SequentialSampler
        batch_sampler = BatchSampler(batch_size=3, drop_last=True, sampler=SequentialSampler([3, 9, 10, 5, 7, 6, 1]))
        result = list(batch_sampler)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_alias_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=torch.utils.data.SequentialSampler([3, 9, 10, 5, 7, 6, 1]), batch_size=3, drop_last=True)
        result = list(batch_sampler)
        """
    )
    obj.run(pytorch_code, ["result"])
