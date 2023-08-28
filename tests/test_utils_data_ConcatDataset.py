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

obj = APIBase("torch.utils.data.ConcatDataset")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import Dataset, ConcatDataset
        class RandomDataset(Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([5]).astype('float32')
                label = np.random.randint(0, 9, (1, )).astype('int64')
                return image, label

            def __len__(self):
                return self.num_samples

        dataset = ConcatDataset([d1, d2])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import Dataset, ConcatDataset
        class RandomDataset(Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([5]).astype('float32')
                label = np.random.randint(0, 9, (1, )).astype('int64')
                return image, label

            def __len__(self):
                return self.num_samples

        dataset = ConcatDataset(datasets=[RandomDataset(2), RandomDataset(2)])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
