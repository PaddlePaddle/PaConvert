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


class DLPackAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        if type(paddle_result).__name__ == "PyCapsule":
            return True
        return False


obj = DLPackAPIBase("torch.utils.dlpack.to_dlpack")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(4).int()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(4).long()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.randn(4, 2).half()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.randn(4, 2).double()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.randn(3, 3).float()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.randn(3, 3).short()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.randn(3, 3).byte()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.randn(3, 3).char()
        result = torch.utils.dlpack.to_dlpack(t)
        """
    )
    obj.run(pytorch_code, ["result"])
