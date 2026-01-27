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

obj = APIBase("torch.utils.data.Dataset")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import Dataset

        class Data(Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = data.__len__()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils.data as data

        class Data(data.Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        my_data = Data()
        result = my_data.__getitem__(0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils as utils

        class Data(utils.data.Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = []
        for i in data:
            result.append(i)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class Data(torch.utils.data.Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = data.__len__()
        """
    )
    obj.run(pytorch_code, ["result"])
