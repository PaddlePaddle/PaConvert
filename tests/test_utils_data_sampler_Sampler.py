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

obj = APIBase("torch.utils.data.sampler.Sampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import Sampler

        class MySampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        data = [0, 1, 2, 3, 4]
        s = MySampler(data)
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

        class MySampler(sampler.Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        my_data = [10, 20, 30]
        s = MySampler(data_source=my_data)
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.utils.data.sampler as sampler

        class MySampler(sampler.Sampler):
            def __init__(self, data):
                self.data_source = data

            def __iter__(self):
                return iter(range(1, len(self.data_source) + 1))

            def __len__(self):
                return len(self.data_source)

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = MySampler(data)
        result = []
        for idx in s:
            result.append(idx)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class MySampler(torch.utils.data.sampler.Sampler):
            def __init__(self, data):
                self.data_source = data

            def __iter__(self):
                return iter(range(1, len(self.data_source) + 1))

            def __len__(self):
                return len(self.data_source)

        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        s = MySampler(data)
        result = []
        for idx in s:
            result.append(idx)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import Sampler

        class MySampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(range(len(self.data_source)))

            def __len__(self):
                return len(self.data_source)

        data = [0, 1, 2, 3, 4]
        s = MySampler(data)
        result = isinstance(s, Sampler)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Mixed arguments test with custom sampler"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data.sampler import Sampler

        class MySampler(Sampler):
            def __init__(self, data_source, start_idx=0):
                self.data_source = data_source
                self.start_idx = start_idx

            def __iter__(self):
                return iter(range(self.start_idx, len(self.data_source)))

            def __len__(self):
                return len(self.data_source) - self.start_idx

        data = [0, 1, 2, 3, 4]
        s = MySampler(data, start_idx=2)
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])
