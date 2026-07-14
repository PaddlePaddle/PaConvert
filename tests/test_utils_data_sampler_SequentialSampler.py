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

obj = APIBase("torch.utils.data.sampler.SequentialSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import SequentialSampler

        s = SequentialSampler([0, 1, 2, 3, 4])
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils.data.sampler as sampler

        my_data = [0, 1, 2, 3, 4]
        s = sampler.SequentialSampler(data_source=my_data)
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils.data.sampler as s_short

        my_data = [0, 1, 2, 3, 4]
        s = s_short.SequentialSampler(my_data)
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class MySampler(torch.utils.data.sampler.SequentialSampler):
            def __init__(self, data):
                self.data_source = data

            def __iter__(self):
                return iter(range(1, len(self.data_source) + 1))

            def __len__(self):
                return len(self.data_source)

        dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = MySampler(dataset)
        result = []
        for d in s:
            result.append(d)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import SequentialSampler
        import torch

        data_source = [10, 20, 30, 40, 50]
        s = SequentialSampler(data_source)
        result = []
        for idx in s:
            result.append(idx)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        s = torch.utils.data.sampler.SequentialSampler([5, 6, 7, 8])
        result = len(s)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Keyword arguments test"""
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import SequentialSampler

        data = [10, 20, 30, 40]
        s = SequentialSampler(data_source=data)
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])
