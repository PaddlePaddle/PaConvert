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

import unittest
import textwrap
from apibase import APIBase

class TestIsReal(unittest.TestCase):
    def setUp(self):
        self.obj = APIBase("torch.isreal")

    def test_case_1(self):
        pytorch_code = textwrap.dedent(
            """
            import torch
            src = torch.tensor([1, 1+1j, 2.0, 3+0j])
            result = torch.isreal(src)
            """
        )
        self.obj.run(pytorch_code, ["result"])

    def test_case_2(self):
        # 验证 input -> x 的映射
        pytorch_code = textwrap.dedent(
            """
            import torch
            src = torch.tensor([1.0, 2.0])
            result = torch.isreal(input=src)
            """
        )
        self.obj.run(pytorch_code, ["result"])

    def test_case_3(self):
        # 验证 Tensor 方法
        pytorch_code = textwrap.dedent(
            """
            import torch
            result = torch.tensor([1, 1j]).isreal()
            """
        )
        self.obj.run(pytorch_code, ["result"])

if __name__ == "__main__":
    unittest.main()