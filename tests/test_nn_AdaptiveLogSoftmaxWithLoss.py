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

obj = APIBase("torch.nn.AdaptiveLogSoftmaxWithLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[ 0.9368637 , -0.0361056 , -0.98917043,  0.06605113,  1.5254455 ],
                            [-1.0518035 , -1.0024613 ,  0.18699688, -0.35807893,  0.25628588],
                            [-0.900478  , -0.41495147,  0.84707606, -1.7883497 ,  1.3243382 ]])
        target = torch.tensor([1, 1, 1])
        asfm = torch.nn.AdaptiveLogSoftmaxWithLoss(5, 4, [2])
        out, loss = asfm(input,target)
        """
    )
    obj.run(pytorch_code, ["out", "loss"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[ 0.9368637 , -0.0361056 , -0.98917043,  0.06605113,  1.5254455 ],
                            [-1.0518035 , -1.0024613 ,  0.18699688, -0.35807893,  0.25628588],
                            [-0.900478  , -0.41495147,  0.84707606, -1.7883497 ,  1.3243382 ]])
        target = torch.tensor([1, 1, 1])
        asfm = torch.nn.AdaptiveLogSoftmaxWithLoss(5, 4, [3], div_value=2.0)
        out, loss = asfm(input,target)
        """
    )
    obj.run(pytorch_code, ["out", "loss"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[ 0.9368637 , -0.0361056 , -0.98917043,  0.06605113,  1.5254455 ],
                            [-1.0518035 , -1.0024613 ,  0.18699688, -0.35807893,  0.25628588],
                            [-0.900478  , -0.41495147,  0.84707606, -1.7883497 ,  1.3243382 ]])
        target = torch.tensor([1, 1, 1])
        asfm = torch.nn.AdaptiveLogSoftmaxWithLoss(5, 4, [1], div_value=3.8, head_bias=True)
        out, loss = asfm(input,target)
        """
    )
    obj.run(pytorch_code, ["out", "loss"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[ 0.9368637 , -0.0361056 , -0.98917043,  0.06605113,  1.5254455 ],
                            [-1.0518035 , -1.0024613 ,  0.18699688, -0.35807893,  0.25628588],
                            [-0.900478  , -0.41495147,  0.84707606, -1.7883497 ,  1.3243382 ]])
        target = torch.tensor([1, 1, 1])
        asfm = torch.nn.AdaptiveLogSoftmaxWithLoss(in_features=5, n_classes=8, cutoffs=[5], div_value=3.8, head_bias=True)
        out, loss = asfm(input,target)
        """
    )
    obj.run(pytorch_code, ["out", "loss"], check_value=False)
