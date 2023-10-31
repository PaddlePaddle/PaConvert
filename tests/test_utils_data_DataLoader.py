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

import paddle
from apibase import APIBase


class DataLoaderAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert isinstance(paddle_result, paddle.io.DataLoader)


obj = DataLoaderAPIBase("torch.utils.data.DataLoader")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import Dataset
        import torch
        class Data(Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = torch.utils.data.DataLoader(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import Dataset
        import torch
        class Data(Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = torch.utils.data.DataLoader(dataset=data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import Dataset
        import torch
        class Data(Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = torch.utils.data.DataLoader(dataset=data,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, batch_size=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import Dataset
        import torch
        class Data(Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = torch.utils.data.DataLoader(dataset=data,
                batch_size=1, shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0,
                worker_init_fn=None,
                multiprocessing_context=None,
                generator=None,
                prefetch_factor=None,
                persistent_workers=False,
                pin_memory_device='')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import Dataset
        import torch
        class Data(Dataset):
            def __init__(self):
                self.x = [1,2,3,4]

            def __getitem__(self, idx):
                return self.x[idx]

            def __len__(self):
                return len(self.x)


        data = Data()
        result = torch.utils.data.DataLoader(data,
                1, False, None,
                None, 0, None,
                False, False, 0,
                None,
                None,
                None,
                prefetch_factor=None,
                persistent_workers=False,
                pin_memory_device='')
        """
    )
    obj.run(pytorch_code, ["result"])
