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

obj = APIBase("torch.utils.data.SequentialSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import SequentialSampler
        from torch.utils.data import Dataset
        import numpy as np

        class MyDataset(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)

        s = SequentialSampler(MyDataset())
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils.data as data
        import numpy as np

        class MyDataset(data.Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)

        my_data = MyDataset()
        s = data.SequentialSampler(data_source=my_data)
        result = []
        for d in s:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils as utils
        import numpy as np
        import torch

        class MyDataset(utils.data.Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1).reshape(10, 10)
                self.y = np.arange(0, 10, 1)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return self.x.shape[0]

        class MySampler(utils.data.SequentialSampler):
            def __init__(self, data):
                self.data_source = data

            def __iter__(self):
                return iter(range(1, len(self.data_source)+1))

            def __len__(self):
                return len(self.data_source)

        dataset = MyDataset()
        s = MySampler(dataset)
        result = []
        for d in s:
            result.append(d)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch

        class MyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1).reshape(10, 10)
                self.y = np.arange(0, 10, 1)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return self.x.shape[0]

        class MySampler(torch.utils.data.SequentialSampler):
            def __init__(self, data):
                self.data_source = data

            def __iter__(self):
                return iter(range(1, len(self.data_source)+1))

            def __len__(self):
                return len(self.data_source)

        dataset = MyDataset()
        s = MySampler(dataset)
        result = []
        for d in s:
            result.append(d)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"])
