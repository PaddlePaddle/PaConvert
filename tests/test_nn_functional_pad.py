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

obj = APIBase("torch.nn.functional.pad", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 0, 0, 0, 0, 1, 2, 3], value=1, mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(input=x, pad=[0, 0, 0, 0, 0, 1, 2, 3], value=1, mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(input=x, value=1, pad=[0, 0, 0, 0, 0, 1, 2, 3], mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(input=x, pad=[0, 0, 0, 0, 0, 1, 2, 3], mode='constant', value=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 0, 0, 0, 0, 1, 2, 3], 'constant', 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result1 = F.pad(x, [0, 0, 1, 2], mode='reflect')
        result2 = F.pad(x, [0, 0, 2, 3], mode='replicate')
        result3 = F.pad(x, [0, 0, 2, 3], mode='circular')
        """
    )
    obj.run(pytorch_code, ["result1", "result2", "result3"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 1, 2, 3], value=1, mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 2, 1, 0, 2, 3], value=1, mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 2], value=1, mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(input=x, pad=[0, 2, 1, 0, 2, 3], value=1, mode='constant')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(input=x, value=1, mode='constant', pad=[0, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 0, 0, 1, 2, 1, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 1, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 2, 1, 0, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x_shape = (3, 3, 4, 2)
        x = torch.arange(torch.prod(torch.Tensor(x_shape)), dtype=torch.float32).reshape(x_shape) + 1
        result = F.pad(x, [0, 2])
        """
    )
    obj.run(pytorch_code, ["result"])
