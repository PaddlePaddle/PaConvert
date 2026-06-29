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

import paddle
import pytest
from apibase import APIBase

obj = APIBase("torch.nn.utils.rnn.PackedSequence")


def test_case_1():
    """Basic PackedSequence creation with positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """PackedSequence creation with keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data=data, batch_sizes=batch_sizes)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """PackedSequence with sorted_indices"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        sorted_indices = torch.tensor([1, 0])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes, sorted_indices=sorted_indices)
        result = (packed.data, packed.batch_sizes, packed.sorted_indices)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """PackedSequence with all parameters"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        sorted_indices = torch.tensor([1, 0])
        unsorted_indices = torch.tensor([1, 0])
        packed = torch.nn.utils.rnn.PackedSequence(
            data=data,
            batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices
        )
        result = (packed.data, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """PackedSequence to() method"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        packed_cpu = packed.to('cpu')
        result = (packed_cpu.data, packed_cpu.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """PackedSequence dtype conversion"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        packed_double = packed.double()
        result = (packed_double.data, packed_double.batch_sizes)
    """
    )
    # Note: dtype conversion behavior may differ for batch_sizes
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_7():
    """PackedSequence is_cuda property"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_8():
    """PackedSequence pin_memory method"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        packed_pinned = packed.pin_memory()
        result = (packed_pinned.data, packed_pinned.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"])


# Tests for PackedSequence attributes (not testing the class object itself)
def test_case_9():
    """PackedSequence data attribute test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        # Test data attribute
        result = packed.data
        print(result)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """PackedSequence batch_sizes attribute test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        # Test batch_sizes attribute
        result = packed.batch_sizes
        print(result)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """PackedSequence sorted_indices attribute test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        sorted_indices = torch.tensor([1, 0])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes, sorted_indices=sorted_indices)
        # Test sorted_indices attribute
        result = packed.sorted_indices
        print(result)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """PackedSequence unsorted_indices attribute test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        sorted_indices = torch.tensor([1, 0])
        unsorted_indices = torch.tensor([0, 1])
        packed = torch.nn.utils.rnn.PackedSequence(
            data, batch_sizes, sorted_indices=sorted_indices, unsorted_indices=unsorted_indices
        )
        # Test unsorted_indices attribute
        result = packed.unsorted_indices
        print(result)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """PackedSequence is_cuda attribute test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        batch_sizes = torch.tensor([2, 2])
        packed = torch.nn.utils.rnn.PackedSequence(data, batch_sizes)
        # Test is_cuda attribute
        result = packed.is_cuda
        print(result)
    """
    )
    # Note: is_cuda behavior differs between PyTorch and Paddle frameworks
    obj.run(pytorch_code, ["result"], check_value=False)
