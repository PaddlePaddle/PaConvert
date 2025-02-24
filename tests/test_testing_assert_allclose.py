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

obj = APIBase("torch.testing.assert_allclose")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        torch.testing.assert_allclose(x, x)
        """
    )
    obj.run(pytorch_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        torch.testing.assert_allclose(actual=x, expected=x)
        """
    )
    obj.run(pytorch_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        torch.testing.assert_allclose(x, x, equal_nan=True)
        """
    )
    obj.run(pytorch_code)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x.detach()
        torch.testing.assert_allclose(x, y, equal_nan=True)
        """
    )
    obj.run(pytorch_code)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x.cpu()
        torch.testing.assert_allclose(actual=x, expected=y, rtol=1e-5, atol=1e-8, equal_nan=True, msg="assert_allclose testing message.")
        """
    )
    obj.run(pytorch_code)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x
        torch.testing.assert_allclose(x, y, 1e-5, 1e-8, True, "assert_allclose testing message.")
        """
    )
    obj.run(pytorch_code)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x
        torch.testing.assert_allclose(msg="assert_allclose testing message.", expected=y, rtol=1e-5, atol=1e-8, equal_nan=True, actual=x)
        """
    )
    obj.run(pytorch_code)
