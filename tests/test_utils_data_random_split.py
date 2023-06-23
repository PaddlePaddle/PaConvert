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
        from torch.utils.data import Dataset

        class Data(Dataset):
            def __init__(self):
                self.x = [0,1,2,3,4,5,6,7,8,9]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()

        datasets = torch.utils.data.random_split(data, [3,7])
        result0 = datasets[0].__len__()
        result1 = datasets[1].__len__()
        """
    )
    obj.run(pytorch_code, ["result0", "result1"])


def test_case_2():
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
        result0 = datasets[0].__len__()
        result1 = datasets[1].__len__()
        """
    )
    obj.run(pytorch_code, ["result0", "result1"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        datasets = torch.utils.data.random_split(range(30), [0.4, 0.4, 0.2])
        result0 = datasets[0].__len__()
        result1 = datasets[1].__len__()
        result2 = datasets[2].__len__()
        """
    )
    obj.run(pytorch_code, ["result0", "result1", "result2"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [0.4, 0.4, 0.2]
        data = range(30)
        datasets = torch.utils.data.random_split(data, lengths)
        result0 = datasets[0].__len__()
        result1 = datasets[1].__len__()
        result2 = datasets[2].__len__()
        """
    )
    obj.run(pytorch_code, ["result0", "result1", "result2"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        lengths = [0.4, 0.4, 0.2]
        data = range(30)
        datasets = torch.utils.data.random_split(data, lengths,generator=torch.Generator().manual_seed(42))
        result0 = datasets[0].__len__()
        result1 = datasets[1].__len__()
        result2 = datasets[2].__len__()
        """
    )
    obj.run(pytorch_code, ["result0", "result1", "result2"])
