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


class LossAPIBase(APIBase):
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
        """Compare string reduction values from _Loss objects"""
        assert (
            pytorch_result == paddle_result
        ), "API ({}): paddle result {} != pytorch result {}".format(
            name, paddle_result, pytorch_result
        )


obj = LossAPIBase("torch.nn.modules.loss._Loss")


def test_case_1():
    """Default reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss()
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduction='none')
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduction='sum')
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='mean' keyword"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduction='mean')
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """size_average=True (deprecated)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(size_average=True)
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """size_average=False (deprecated)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(size_average=False)
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """reduce=False (deprecated)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduce=False)
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """reduce=True, size_average=False (deprecated)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduce=True, size_average=False)
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduction='sum', size_average=None, reduce=None)
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """All None deprecated args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(size_average=None, reduce=None, reduction='none')
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Expression reduction argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss._Loss(reduction='sum' if 1 == 1 else 'mean')
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Variable reduction argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        red = 'none'
        loss = torch.nn.modules.loss._Loss(reduction=red)
        result = loss.reduction
        """
    )
    obj.run(pytorch_code, ["result"])
