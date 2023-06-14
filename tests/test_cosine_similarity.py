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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.cosine_similarity")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([[1.4309,  1.2706], [-0.8562,  0.9796]])
        x2 = torch.ones_like(x1)
        result = torch.cosine_similarity(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([[1.4309,  1.2706], [-0.8562,  0.9796]])
        x2 = torch.ones_like(x1)
        result = torch.cosine_similarity(x1=x1, x2=x2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796])
        x2 = torch.ones_like(x1)
        result = torch.cosine_similarity(x1=x1, x2=x2, axis=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1 = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796])
        x2 = torch.ones_like(x1)
        result = torch.cosine_similarity(x1=x1, x2=x2, axis=0, eps=0.1)
        """
    )
    obj.run(pytorch_code, ["result"])
