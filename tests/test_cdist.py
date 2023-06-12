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

obj = APIBase("torch.cdist")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([[ 1.6830,  0.0526],
            [-0.0696,  0.6366],
            [-1.0091,  1.3363]])
        x2 = torch.tensor([[-0.0629,  0.2414],
            [-0.9701, -0.4455]])
        result = torch.cdist(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([[ 1.6830,  0.0526],
            [-0.0696,  0.6366],
            [-1.0091,  1.3363]])
        x2 = torch.tensor([[-0.0629,  0.2414],
            [-0.9701, -0.4455]])
        result = torch.cdist(x1=x1, x2=x2, p=1.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([[[ 1.5518, -1.2166],
                [ 0.2780, -0.5918],
                [-0.6906,  1.0884]],
                [[-0.3519,  0.2204],
                [-1.0994,  0.1239],
                [ 0.4219,  0.0442]]])
        x2 = torch.tensor([[[-0.5764,  0.6476],
                [-0.5335, -0.7144]],
                [[ 0.0617,  0.8019],
                [-0.3107, -0.8516]]])
        result = torch.cdist(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"])
