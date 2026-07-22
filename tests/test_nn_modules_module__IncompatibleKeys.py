# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.modules.module._IncompatibleKeys")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.Linear(1, 2)
        incompatible_keys = model.load_state_dict({"a": 2.0}, strict=False)
        """
    )
    obj.run(pytorch_code, ["incompatible_keys"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.Linear(2, 3)
        missing, unexpected = model.load_state_dict({"b": -2.0}, strict=False)
        """
    )
    obj.run(pytorch_code, ["missing", "unexpected"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.Linear(6, 2)
        result = model.load_state_dict({"c": -6.0}, strict=False)
        result_0 = result[0]
        result_1 = result[1]
        result_missing = result.missing_keys
        result_unexpected = result.unexpected_keys
        """
    )
    obj.run(
        pytorch_code, ["result_0", "result_1", "result_missing", "result_unexpected"]
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from collections import namedtuple
        model = torch.nn.Linear(6, 2)
        result = model.load_state_dict({"c": -6.0}, strict=False)
        is_tuple = isinstance(result, tuple)
        """
    )
    obj.run(pytorch_code, ["is_tuple"])
