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

obj = APIBase("torch.Tensor.cdouble")


def test_uint8_case():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([0., 1., 2., 3.,], dtype=torch.uint8)
        result = src.cdouble()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case():
    text_code = """
        import torch
        src = torch.tensor([0., -1., 2.34, -3.45, -0.34, 0.23, 1., 2., 3.,], dtype=torch.bfloat16).cuda()
        result = src.cdouble()
        """
    dtypes = [
        "torch.float16",
        "torch.float32",
        "torch.float64",
        "torch.bfloat16",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.bool",
        "torch.complex64",
        "torch.complex128",
    ]

    for dtype in dtypes:
        text_code_test = text_code.replace("torch.bfloat16", dtype)
        pytorch_code = textwrap.dedent(text_code_test)
        obj.run(pytorch_code, ["result"])
