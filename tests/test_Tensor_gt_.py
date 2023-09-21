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

obj = APIBase("torch.Tensor.gt_")


# shield unittests for unexpected behaviour of `Tensor.cast_`
# ref: https://github.com/PaddlePaddle/PaConvert/pull/294
def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        x.gt_(torch.tensor([[1, 1], [4, 4]]))
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        x.gt_(other=torch.tensor([[1, 1], [4, 4]]))
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 1], [4, 4]])
        x.gt_(other)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 1], [4, 4]])
        x.gt_(other=other)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 2], [3, 4]])
        x.gt_(other)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 2], [3, 4]])
        x.gt_(other=other)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([1, 2])
        x.gt_(other)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([1, 2])
        x.gt_(other=other)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        x.gt_(2)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        x.gt_(other=2)
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 2.], [3., 4.]])
        x.gt_(torch.tensor([[1., 1.], [4., 4.]]))
        """
    )
    obj.run(pytorch_code, ["x"])


def _test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 2.], [3., 4.]])
        x.gt_(2.)
        """
    )
    obj.run(pytorch_code, ["x"])
