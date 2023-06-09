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
        result = list(BatchSampler(range(10), batch_size=3, drop_last=True))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import BatchSampler
        result = list(BatchSampler(range(10), batch_size=3, drop_last=False))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = batch_sampler_train = torch.utils.data.BatchSampler(range(10), 2, drop_last=True)
        result = list(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        batch_size = 4
        result = batch_sampler_train = torch.utils.data.BatchSampler(range(10), batch_size, drop_last=False)
        result = list(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        batch_size = 4
        result = list(torch.utils.data.BatchSampler(sampler=range(10), batch_size=batch_size, drop_last=False))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_alias_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        batch_size = 4
        result = list(torch.utils.data.sampler.BatchSampler(sampler=range(10), batch_size=batch_size, drop_last=False))
        """
    )
    obj.run(pytorch_code, ["result"])
