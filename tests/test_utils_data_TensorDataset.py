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

obj = APIBase("torch.utils.data.TensorDataset")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        np.random.seed(0)
        input_np = np.random.random([2, 3, 4]).astype('float32')
        input = torch.from_numpy(input_np)
        dataset = TensorDataset(input)
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        np.random.seed(0)
        input_np = np.random.random([2, 3, 4]).astype('float32')
        input = torch.from_numpy(input_np)
        label_np = np.random.random([2, 1]).astype('int32')
        label = torch.from_numpy(label_np)
        dataset = TensorDataset(input, label)
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        np.random.seed(0)
        input_np = np.random.random([2, 3, 4]).astype('float32')
        input = torch.from_numpy(input_np)
        input_np2 = np.random.random([2, 5, 5]).astype('float32')
        input2 = torch.from_numpy(input_np2)
        label_np = np.random.random([2, 1]).astype('int32')
        label = torch.from_numpy(label_np)
        dataset = TensorDataset(input, input2, label)
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        np.random.seed(0)
        input_np = np.random.random([2, 3, 4]).astype('float32')
        input = torch.from_numpy(input_np)
        input_np2 = np.random.random([2, 5, 5]).astype('float32')
        input2 = torch.from_numpy(input_np2)
        label_np = np.random.random([2, 1]).astype('int32')
        label = torch.from_numpy(label_np)
        data = [input, input2, label]

        dataset = TensorDataset(*data)
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        np.random.seed(0)
        input_np = np.random.random([2, 3, 4]).astype('float32')
        input = torch.from_numpy(input_np)
        input_np2 = np.random.random([2, 5, 5]).astype('float32')
        input2 = torch.from_numpy(input_np2)
        label_np = np.random.random([2, 1]).astype('int32')
        label = torch.from_numpy(label_np)
        dataset = TensorDataset(input=input, input2=input2, label=label)
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        from torch.utils.data import TensorDataset
        np.random.seed(0)
        input_np = np.random.random([2, 3, 4]).astype('float32')
        input = torch.from_numpy(input_np)
        input_np2 = np.random.random([2, 5, 5]).astype('float32')
        input2 = torch.from_numpy(input_np2)
        label_np = np.random.random([2, 1]).astype('int32')
        label = torch.from_numpy(label_np)
        dataset = TensorDataset(input=input, label=label, input2=input2)
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])
