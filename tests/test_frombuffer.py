# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.frombuffer", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('i', [1, 2, 3])
        result = torch.frombuffer(a, dtype=torch.int32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a =array.array('b', [-1, 0, 0, 0])
        result = torch.frombuffer(a, dtype=torch.int32)
        """
    )
    obj.run(pytorch_code, ["result"])
