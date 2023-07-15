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

obj = APIBase("torch.Tensor.absolute")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[-4, 9], [-23, 2]])
        result = a.absolute()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([[-4, 9], [-23, 2]]).absolute()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        try:
            a = torch.tensor([[-4, 9], [-23, 2]])
            assert 0, "Raise AssertionError"
        except Exception as e:
            error_msg = str(e)
        """
    )
    obj.run(pytorch_code, ["error_msg"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        try:
            a = torch.tensor([[-4, 9], [-23, 2]])
            assert 0, "Raise AssertionError"
        except Exception as e:
            error_msg = str(e)
        finally:
            pass
        """
    )
    obj.run(pytorch_code, ["error_msg"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        i = 0
        result = []
        while i < 5:
            result.append(torch.tensor(i).absolute())
            i += 1
        """
    )
    obj.run(pytorch_code, ["result"])
