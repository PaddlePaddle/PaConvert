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

obj = APIBase("torch.Tensor.tril_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                        [ 0.0935,  0.1380,  2.2112],
                        [-0.3409, -0.9828,  0.0289]])
        x.tril_()
        """
    )
    obj.run(pytorch_code, ["x"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                        [ 0.0935,  0.1380,  2.2112],
                        [-0.3409, -0.9828,  0.0289]])
        x.tril_(1)
        """
    )
    obj.run(pytorch_code, ["x"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                        [ 0.0935,  0.1380,  2.2112],
                        [-0.3409, -0.9828,  0.0289]])
        x.tril_(diagonal=-1)
        """
    )
    obj.run(pytorch_code, ["x"])
