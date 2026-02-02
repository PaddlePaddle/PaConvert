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

obj = APIBase("torch.utils.data.random_split")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class Data(torch.utils.data.Dataset):
            def __init__(self):
                self.x = [0,1,2,3,4,5,6,7,8,9]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()

        datasets = torch.utils.data.random_split(data, [3, 7])

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils.data as data

        class Data(data.Dataset):
            def __init__(self):
                self.x = [0,1,2,3,4,5,6,7,8,9]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        my_data = Data()
        datasets = data.random_split(my_data, [3, 3, 4])

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils as utils

        class Data(utils.data.Dataset):
            def __init__(self):
                self.x = [0,1,2,3,4,5,6,7,8,9]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        lengths = [3, 3, 4]
        datasets = utils.data.random_split(data, lengths)

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data import Dataset

        class Data(Dataset):
            def __init__(self):
                self.x = [0,1,2,3,4,5,6,7,8,9]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        lengths = [0.4, 0.4, 0.2]
        datasets = torch.utils.data.random_split(data, lengths)

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import random_split

        datasets = random_split(range(30), [0.4, 0.4, 0.2])

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [0.4, 0.4, 0.2]
        data = range(30)
        datasets = torch.utils.data.random_split(data, lengths)

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [0.4, 0.4, 0.2]
        data = range(30)
        datasets = torch.utils.data.random_split(data, lengths,generator=torch.Generator().manual_seed(42))

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [3, 3, 4]
        data = range(10)
        datasets = torch.utils.data.random_split(dataset=data, lengths=lengths, generator=torch.Generator().manual_seed(42))

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [3, 3, 4]
        data = range(10)
        datasets = torch.utils.data.random_split(data, lengths, torch.Generator().manual_seed(42))

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [3, 3, 4]
        data = range(10)
        datasets = torch.utils.data.random_split(lengths=lengths, dataset=data, generator=torch.Generator().manual_seed(42))

        results = []
        for d in datasets:
            results.append(d.__len__())
        """
    )
    obj.run(pytorch_code, ["results"])
