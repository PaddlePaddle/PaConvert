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

obj = APIBase("torch.utils.data.sampler.RandomSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import RandomSampler
        import torch

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = RandomSampler(data, num_samples=10, replacement=False)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import RandomSampler
        import torch

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = RandomSampler(data_source=data, num_samples=10, replacement=True)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import RandomSampler
        import torch

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = RandomSampler(data, num_samples=None, replacement=True)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import RandomSampler
        import torch

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = RandomSampler(data, False, 10)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import RandomSampler
        import torch

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = RandomSampler(data_source=data, replacement=True, num_samples=3)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.utils.data.sampler as sampler

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = sampler.RandomSampler(data, True, 3)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import RandomSampler
        import torch

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = RandomSampler(num_samples=5, data_source=data, replacement=True)
        result = []
        for idx, val in enumerate(s):
            result.append(val)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
