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

obj = APIBase("torch.distributions.CatTransform")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
        x = torch.cat([x0, x0], dim=0)
        t0 = torch.distributions.CatTransform([torch.distributions.ExpTransform(), torch.distributions.ExpTransform()], dim=0, lengths=[10, 10])
        t = torch.distributions.CatTransform([t0, t0], dim=0, lengths=[20, 20])
        result = t(x)
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
        x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
        x = torch.cat([x0, x0], dim=0)
        t0 = torch.distributions.CatTransform([torch.distributions.ExpTransform(), torch.distributions.ExpTransform()], dim=0, lengths=[10, 10])
        t = torch.distributions.CatTransform(tseq=[t0, t0], dim=0, lengths=[20, 20])
        result = t(x)
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
        x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
        x = torch.cat([x0, x0], dim=0)
        t0 = torch.distributions.CatTransform([torch.distributions.ExpTransform(), torch.distributions.ExpTransform()], dim=0, lengths=[10, 10])
        t = torch.distributions.CatTransform(tseq=[t0, t0], lengths=[20, 20], dim=0)
        result = t(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
