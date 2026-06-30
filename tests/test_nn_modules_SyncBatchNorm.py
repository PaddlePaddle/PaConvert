# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.modules.SyncBatchNorm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.SyncBatchNorm(10)
        result = model(torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.SyncBatchNorm(10, eps=1e-5)
        result = model(torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.SyncBatchNorm(num_features=10, affine=True)
        result = model(torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.SyncBatchNorm(num_features=10, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        result = model(torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
