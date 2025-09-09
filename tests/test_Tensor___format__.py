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
        if not isinstance(paddle_result, str):
            paddle_result = str(paddle_result)
        if not isinstance(pytorch_result, str):
            pytorch_result = str(pytorch_result)

        start_idx = pytorch_result.find("(")
        end_idx = pytorch_result.rfind(")")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            torch_content = pytorch_result[start_idx + 1 : end_idx].strip()
            if torch_content not in paddle_result:
                raise AssertionError(
                    f"API ({name}): Paddle result does not contain the core content from PyTorch result.\n"
                    f"Core content from PyTorch: '{torch_content}'\n"
                    f"Full Pytorch result: '{pytorch_result}'\n"
                    f"Full Paddle result: '{paddle_result}'"
                )
        else:
            assert pytorch_result == paddle_result, (
                f"API ({name}): Paddle result does not contain PyTorch result.\n"
                f"Pytorch result: '{pytorch_result}'\n"
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
