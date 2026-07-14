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

obj = APIBase("torch.utils.data.dataset.ConcatDataset")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data.dataset import ConcatDataset
        dataset1 = [1., 2., 3.]
        dataset2 = [4., 5.]
        dataset = ConcatDataset([dataset1, dataset2])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data.dataset import ConcatDataset as Concat
        dataset1 = [1., 2., 3.]
        dataset2 = [4., 5.]
        dataset = Concat(datasets=[dataset1, dataset2])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dataset1 = [1., 2., 3.]
        dataset2 = [4., 5.]
        dataset = torch.utils.data.dataset.ConcatDataset(datasets=[dataset1, dataset2])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.utils.data.dataset as ds
        dataset1 = [1., 2., 3.]
        dataset2 = [4., 5.]
        dataset = ds.ConcatDataset([dataset1, dataset2])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.utils.data.dataset as ds_short
        dataset1 = [1., 2., 3.]
        dataset2 = [4., 5.]
        dataset = ds_short.ConcatDataset([dataset1, dataset2])
        result = []
        for i in range(len(dataset)):
            result.append(dataset[i])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.utils.data.dataset import ConcatDataset
        dataset1 = [1., 2., 3.]
        dataset2 = [4., 5., 6.]
        dataset = ConcatDataset(datasets=[dataset1, dataset2])
        result = len(dataset)
        """
    )
    obj.run(pytorch_code, ["result"])
