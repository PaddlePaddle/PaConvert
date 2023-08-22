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

obj = APIBase("torch.nn.LazyInstanceNorm2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.LazyInstanceNorm2d()
        input = torch.tensor([[[[0.9436, 0.7335, 0.9228],
          [0.5443, 0.3380, 0.0676],
          [0.2152, 0.2725, 0.2988]],

         [[0.3839, 0.7517, 0.8147],
          [0.7681, 0.0924, 0.3781],
          [0.6991, 0.2401, 0.4732]],

         [[0.3631, 0.5113, 0.4535],
          [0.9779, 0.4084, 0.5979],
          [0.6865, 0.5924, 0.9122]]],


        [[[0.1519, 0.2828, 0.0797],
          [0.5871, 0.1052, 0.2343],
          [0.0323, 0.0754, 0.6707]],

         [[0.6969, 0.4170, 0.0762],
          [0.2514, 0.5124, 0.3972],
          [0.1007, 0.7754, 0.4779]],

         [[0.1753, 0.2245, 0.0369],
          [0.5224, 0.9840, 0.0497],
          [0.8938, 0.5135, 0.5939]]]])
        result = m(input)
        result.requires_grad = False
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.LazyInstanceNorm2d(affine=True)
        input = torch.tensor([[[[0.9436, 0.7335, 0.9228],
          [0.5443, 0.3380, 0.0676],
          [0.2152, 0.2725, 0.2988]],

         [[0.3839, 0.7517, 0.8147],
          [0.7681, 0.0924, 0.3781],
          [0.6991, 0.2401, 0.4732]],

         [[0.3631, 0.5113, 0.4535],
          [0.9779, 0.4084, 0.5979],
          [0.6865, 0.5924, 0.9122]]],


        [[[0.1519, 0.2828, 0.0797],
          [0.5871, 0.1052, 0.2343],
          [0.0323, 0.0754, 0.6707]],

         [[0.6969, 0.4170, 0.0762],
          [0.2514, 0.5124, 0.3972],
          [0.1007, 0.7754, 0.4779]],

         [[0.1753, 0.2245, 0.0369],
          [0.5224, 0.9840, 0.0497],
          [0.8938, 0.5135, 0.5939]]]])
        result = m(input)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.LazyInstanceNorm2d(affine=False)
        input = torch.tensor([[[[0.9436, 0.7335, 0.9228],
          [0.5443, 0.3380, 0.0676],
          [0.2152, 0.2725, 0.2988]],

         [[0.3839, 0.7517, 0.8147],
          [0.7681, 0.0924, 0.3781],
          [0.6991, 0.2401, 0.4732]],

         [[0.3631, 0.5113, 0.4535],
          [0.9779, 0.4084, 0.5979],
          [0.6865, 0.5924, 0.9122]]],


        [[[0.1519, 0.2828, 0.0797],
          [0.5871, 0.1052, 0.2343],
          [0.0323, 0.0754, 0.6707]],

         [[0.6969, 0.4170, 0.0762],
          [0.2514, 0.5124, 0.3972],
          [0.1007, 0.7754, 0.4779]],

         [[0.1753, 0.2245, 0.0369],
          [0.5224, 0.9840, 0.0497],
          [0.8938, 0.5135, 0.5939]]]])
        result = m(input)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.LazyInstanceNorm2d(affine=True, momentum=0.1)
        input = torch.tensor([[[[0.9436, 0.7335, 0.9228],
          [0.5443, 0.3380, 0.0676],
          [0.2152, 0.2725, 0.2988]],

         [[0.3839, 0.7517, 0.8147],
          [0.7681, 0.0924, 0.3781],
          [0.6991, 0.2401, 0.4732]],

         [[0.3631, 0.5113, 0.4535],
          [0.9779, 0.4084, 0.5979],
          [0.6865, 0.5924, 0.9122]]],


        [[[0.1519, 0.2828, 0.0797],
          [0.5871, 0.1052, 0.2343],
          [0.0323, 0.0754, 0.6707]],

         [[0.6969, 0.4170, 0.0762],
          [0.2514, 0.5124, 0.3972],
          [0.1007, 0.7754, 0.4779]],

         [[0.1753, 0.2245, 0.0369],
          [0.5224, 0.9840, 0.0497],
          [0.8938, 0.5135, 0.5939]]]])
        result = m(input)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.LazyInstanceNorm2d(affine=False, momentum=0.1)
        input = torch.tensor([[[[0.9436, 0.7335, 0.9228],
          [0.5443, 0.3380, 0.0676],
          [0.2152, 0.2725, 0.2988]],

         [[0.3839, 0.7517, 0.8147],
          [0.7681, 0.0924, 0.3781],
          [0.6991, 0.2401, 0.4732]],

         [[0.3631, 0.5113, 0.4535],
          [0.9779, 0.4084, 0.5979],
          [0.6865, 0.5924, 0.9122]]],


        [[[0.1519, 0.2828, 0.0797],
          [0.5871, 0.1052, 0.2343],
          [0.0323, 0.0754, 0.6707]],

         [[0.6969, 0.4170, 0.0762],
          [0.2514, 0.5124, 0.3972],
          [0.1007, 0.7754, 0.4779]],

         [[0.1753, 0.2245, 0.0369],
          [0.5224, 0.9840, 0.0497],
          [0.8938, 0.5135, 0.5939]]]])
        result = m(input)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.LazyInstanceNorm2d(affine=False, momentum=0.1, dtype=torch.float32)
        input = torch.tensor([[[[0.9436, 0.7335, 0.9228],
          [0.5443, 0.3380, 0.0676],
          [0.2152, 0.2725, 0.2988]],

         [[0.3839, 0.7517, 0.8147],
          [0.7681, 0.0924, 0.3781],
          [0.6991, 0.2401, 0.4732]],

         [[0.3631, 0.5113, 0.4535],
          [0.9779, 0.4084, 0.5979],
          [0.6865, 0.5924, 0.9122]]],


        [[[0.1519, 0.2828, 0.0797],
          [0.5871, 0.1052, 0.2343],
          [0.0323, 0.0754, 0.6707]],

         [[0.6969, 0.4170, 0.0762],
          [0.2514, 0.5124, 0.3972],
          [0.1007, 0.7754, 0.4779]],

         [[0.1753, 0.2245, 0.0369],
          [0.5224, 0.9840, 0.0497],
          [0.8938, 0.5135, 0.5939]]]])
        result = m(input)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
