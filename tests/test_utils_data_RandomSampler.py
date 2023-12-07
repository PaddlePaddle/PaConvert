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

obj = APIBase("torch.utils.data.RandomSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data, num_samples=10, replacement=False)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle does not support assign num_samples when replacement is False",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data, num_samples=10, replacement=True)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data, num_samples=None, replacement=True)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data, num_samples=None, replacement=False)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data, False, 10)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle does not support assign num_samples when replacement is False",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        s = RandomSampler(data, True, 3)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        g = torch.Generator()
        s = RandomSampler(data, True, 3, g)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import RandomSampler, Dataset
        import torch
        import numpy as np

        class Data(Dataset):
            def __init__(self):
                self.x = np.arange(0, 100, 1)

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return self.x.shape[0]

        data = Data()
        g = torch.Generator()
        s = RandomSampler(data_source=data, replacement=True, num_samples=3, generator=g)
        result = []
        for idx, data in enumerate(s):
            result.append(data)
        result = torch.tensor(result)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
