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

obj = APIBase("torch.nn.modules.CosineSimilarity")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity()
        result = model(torch.randn(3, 10), torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(dim=1)
        result = model(torch.randn(3, 10), torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(dim=1, eps=1e-8)
        result = model(torch.randn(3, 10), torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(eps=1e-8)
        result = model(torch.randn(3, 10), torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(dim=0)
        x1 = torch.randn(10, 3)
        x2 = torch.randn(10, 3)
        result = model(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(dim=1, eps=1e-6)
        x1 = torch.randn(3, 10)
        x2 = torch.randn(3, 10)
        result = model(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity()
        x1 = torch.randn(3, 5, 10)
        x2 = torch.randn(3, 5, 10)
        result = model(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(dim=2)
        x1 = torch.randn(3, 5, 10)
        x2 = torch.randn(3, 5, 10)
        result = model(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(dim=0, eps=1e-6)
        x1 = torch.randn(10, 5)
        x2 = torch.randn(10, 5)
        result = model(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity()
        x1 = torch.randn(4, 8)
        x2 = torch.randn(4, 8)
        result = model(x1, x2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    """Mixed arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(1, eps=1e-8)
        result = model(torch.randn(3, 10), torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.CosineSimilarity(eps=1e-6, dim=1)
        result = model(torch.randn(3, 10), torch.randn(3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
