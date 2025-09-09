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


class TensorFormatAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):

        assert isinstance(
            pytorch_result, str
        ), f"API ({name}): The return value must be string type."

        assert isinstance(
            paddle_result, str
        ), f"API ({name}): The return value must be string type."

        def extract_last_bracket_content(s):
            start = s.rfind("(")
            end = s.rfind(")")
            if start != -1 and end != -1 and end > start:
                return s[start + 1 : end].strip()
            return s

        torch_content = extract_last_bracket_content(pytorch_result)

        if torch_content not in paddle_result:
            raise AssertionError(
                f"API ({name}): Content in last brackets mismatch.\n"
                f"PyTorch result: '{pytorch_result}'\n"
                f"Paddle result: '{paddle_result}'"
            )


obj = TensorFormatAPIBase("torch.Tensor.__format__")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(3.14159)
        result = format(x, '.2f')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(123)
        result = format(x, '05d')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        result = format(x, '')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        result = "{}".format(x)
        """
    )
    obj.run(pytorch_code, ["result"])
