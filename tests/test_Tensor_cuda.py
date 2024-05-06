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

obj = APIBase("torch.Tensor.cuda")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1,2,3])
        result = None
        if torch.cuda.is_available():
            result = a.cuda()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = None
        if torch.cuda.is_available():
            result = a.cuda(device="cuda:0", non_blocking=True, memory_format=torch.channels_last)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = None
        if torch.cuda.is_available():
            result = a.cuda("cuda:0", True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = None
        if torch.cuda.is_available():
            result = a.cuda(non_blocking=True, device="cuda:0", memory_format=torch.channels_last)
        """
    )
    obj.run(pytorch_code, ["result"])
