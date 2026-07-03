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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.nn.utils.rnn.invert_permutation")


def test_case_1():
    """Basic invert_permutation with positional argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([2, 0, 1])
        inv_perm = torch.nn.utils.rnn.invert_permutation(perm)
        print(inv_perm)
    """
    )
    obj.run(pytorch_code, ["inv_perm"])


def test_case_2():
    """invert_permutation with keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([2, 0, 1])
        inv_perm = torch.nn.utils.rnn.invert_permutation(permutation=perm)
        print(inv_perm)
    """
    )
    obj.run(pytorch_code, ["inv_perm"])


def test_case_3():
    """invert_permutation with None input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.utils.rnn.invert_permutation(None)
        print(result)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """invert_permutation identity permutation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([0, 1, 2, 3])
        inv_perm = torch.nn.utils.rnn.invert_permutation(perm)
        print(inv_perm)
    """
    )
    obj.run(pytorch_code, ["inv_perm"])


def test_case_5():
    """invert_permutation reverse permutation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([3, 2, 1, 0])
        inv_perm = torch.nn.utils.rnn.invert_permutation(perm)
        print(inv_perm)
    """
    )
    obj.run(pytorch_code, ["inv_perm"])


def test_case_6():
    """invert_permutation with different dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([2, 0, 1], dtype=torch.int64)
        inv_perm = torch.nn.utils.rnn.invert_permutation(perm)
        print(inv_perm)
    """
    )
    obj.run(pytorch_code, ["inv_perm"])


def test_case_7():
    """invert_permutation verification"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([4, 1, 3, 0, 2])
        inv_perm = torch.nn.utils.rnn.invert_permutation(perm)
        # Verify: perm[inv_perm] should be [0, 1, 2, 3, 4]
        result = perm[inv_perm]
        print(result)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        perm = torch.tensor([2, 0, 1], dtype=torch.float32, requires_grad=True)
        # Create an output that depends on the permutation
        values = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
        result = values[perm.long()]
        result.sum().backward()
        perm_grad = perm.grad
        values_grad = values.grad
    """
    )
    obj.run(
        pytorch_code, ["result", "perm_grad", "values_grad"], check_stop_gradient=False
    )


def test_case_9():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        # Use expression as permutation input
        base = torch.tensor([0, 1, 2])
        perm = base + 2  # [2, 3, 4] -> not valid permutation
        # Use a valid permutation expression
        perm = torch.tensor([2, 0, 1]) * 1
        inv_perm = torch.nn.utils.rnn.invert_permutation(perm)
        print(inv_perm)
    """
    )
    obj.run(pytorch_code, ["inv_perm"])
