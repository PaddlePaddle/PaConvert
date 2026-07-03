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

import pytest
from apibase import APIBase

obj = APIBase("torch.linalg.pinv")


def test_case_1():
    """Basic usage - float64 input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape((3, 5)).to(dtype=torch.float64)
        result = torch.linalg.pinv(x)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_2():
    """Keyword argument - input keyword"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape((3, 5)).to(dtype=torch.float64)
        result = torch.linalg.pinv(input=x)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_3():
    """Hermitian matrix with out parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [2, 1]]).to(dtype=torch.float32)
        out = torch.tensor([])
        result = torch.linalg.pinv(hermitian=True, input=x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_4():
    """All keyword arguments - out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [2, 1]]).to(dtype=torch.float32)
        out = torch.tensor([])
        result = torch.linalg.pinv(input=x, atol=None, rtol=1e-5, hermitian=False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_5():
    """Mixed positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [2, 1]]).to(dtype=torch.float32)
        out = torch.tensor([])
        result = torch.linalg.pinv(x, atol=None, rtol=1e-5, hermitian=False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    """Float32 input - using deterministic data for consistent comparison"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32)
        result = torch.linalg.pinv(x, rtol=1e-5)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_7():
    """Batched input - 3D tensor with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                          [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]]], dtype=torch.float64)
        result = torch.linalg.pinv(x, rtol=1e-8)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_8():
    """With explicit atol value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.pinv(x, atol=1e-6)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_9():
    """With explicit rtol value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.pinv(x, rtol=1e-5)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_10():
    """With both atol and rtol"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        result = torch.linalg.pinv(x, atol=1e-10, rtol=1e-8)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_11():
    """Hermitian matrix with float64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        result = torch.linalg.pinv(x, hermitian=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_12():
    """Expression argument - atol as expression"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.pinv(x, atol=1e-3 * 1e-3)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_13():
    """Square matrix input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], dtype=torch.float64)
        result = torch.linalg.pinv(x)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_14():
    """Tall matrix (m > n) with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=torch.float64)
        result = torch.linalg.pinv(x, rtol=1e-8)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_15():
    """Wide matrix (m < n) with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=torch.float64)
        result = torch.linalg.pinv(x, rtol=1e-8)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_16():
    """Batched Hermitian matrix with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[2.0, 1.0, 0.5], [1.0, 3.0, 1.0], [0.5, 1.0, 2.0]],
                          [[3.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 1.0]]], dtype=torch.float64)
        result = torch.linalg.pinv(x, hermitian=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_17():
    """With out parameter for float32"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        out = torch.empty(2, 3, dtype=torch.float32)
        result = torch.linalg.pinv(x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"], atol=1e-5)


def test_case_18():
    """Tensor type atol"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        atol_tensor = torch.tensor(1e-10)
        result = torch.linalg.pinv(x, atol=atol_tensor)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_19():
    """Tensor type rtol"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        rtol_tensor = torch.tensor(1e-8)
        result = torch.linalg.pinv(x, rtol=rtol_tensor)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_20():
    """Complex input - Hermitian matrix (complex64) with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[2.0+0j, 1.0+1j, 0.5-0.5j], [1.0-1j, 3.0+0j, 1.0+0j], [0.5+0.5j, 1.0+0j, 2.0+0j]], dtype=torch.complex64)
        result = torch.linalg.pinv(x, hermitian=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_21():
    """Complex input - Hermitian matrix (complex128) with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[2.0+0j, 1.0+1j, 0.5-0.5j], [1.0-1j, 3.0+0j, 1.0+0j], [0.5+0.5j, 1.0+0j, 2.0+0j]], dtype=torch.complex128)
        result = torch.linalg.pinv(x, hermitian=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


@pytest.mark.skip(
    reason="Paddle's maximum kernel does not support complex types for non-Hermitian pinv with atol/rtol"
)
def test_case_22():
    """Complex input - non-Hermitian matrix with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0+1j, 2.0+0j, 3.0-1j], [4.0-0.5j, 5.0+0.5j, 6.0+0j]], dtype=torch.complex128)
        result = torch.linalg.pinv(x, rtol=1e-6)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_23():
    """Keyword arguments completely out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        result = torch.linalg.pinv(rtol=1e-8, atol=1e-10, input=x)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_24():
    """Verify pinv property: A @ pinv(A) @ A ≈ A with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=torch.float64)
        pinv_A = torch.linalg.pinv(A, rtol=1e-8)
        result = A @ pinv_A @ A
        expected = A
        """
    )
    obj.run(pytorch_code, ["result", "expected"], atol=1e-7)


def test_case_25():
    """Verify pinv property: pinv(A) @ A @ pinv(A) ≈ pinv(A) with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=torch.float64)
        pinv_A = torch.linalg.pinv(A, rtol=1e-8)
        result = pinv_A @ A @ pinv_A
        expected = pinv_A
        """
    )
    obj.run(pytorch_code, ["result", "expected"], atol=1e-7)


def test_case_26():
    """Float32 input default behavior with deterministic data and explicit rtol"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32)
        result = torch.linalg.pinv(x, rtol=1e-5)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_27():
    """Rank-deficient matrix with appropriate tolerance"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        # Create a rank-deficient matrix
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float64)
        result = torch.linalg.pinv(x, rtol=1e-5)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_28():
    """Batched input with varying batch dimensions - deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0]],
                          [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0], [18.0, 19.0, 20.0, 21.0]],
                          [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0], [19.0, 20.0, 21.0, 22.0]]],
                         [[[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
                          [[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]],
                          [[6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0], [18.0, 19.0, 20.0, 21.0], [22.0, 23.0, 24.0, 25.0]]]], dtype=torch.float64)
        result = torch.linalg.pinv(x, rtol=1e-8)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_29():
    """A parameter alias test with deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=torch.float64)
        result = torch.linalg.pinv(A, rtol=1e-8)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_30():
    """Large tolerance test for numerical stability - deterministic data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0, 26.0]], dtype=torch.float64)
        # Use larger tolerance to filter small singular values
        result = torch.linalg.pinv(x, atol=0.1, rtol=0.01)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)
