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

obj = APIBase("torch.linalg.cholesky_ex")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.1481, 0.9974, 0.9413],
                [0.9974, 1.3924, 0.6773],
                [0.9413, 0.6773, 1.1315]])
        result = torch.linalg.cholesky_ex(a)
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
        import torch
        a = torch.tensor([[1.1481, 0.9974, 0.9413],
                [0.9974, 1.3924, 0.6773],
                [0.9413, 0.6773, 1.1315]])
        result = torch.linalg.cholesky_ex(a, upper=False)
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
        import torch
        a = torch.tensor([[1.1481, 0.9974, 0.9413],
                [0.9974, 1.3924, 0.6773],
                [0.9413, 0.6773, 1.1315]])
        out = torch.randn(3, 3)
        result = torch.linalg.cholesky_ex(a, upper=True, out=out)
        """
    )
    obj.run(
        pytorch_code,
        ["result", "out"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
