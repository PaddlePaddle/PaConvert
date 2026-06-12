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

obj = APIBase("torch.nn.utils.rnn.unpack_sequence")


def test_case_1():
    """Basic unpack_sequence with positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Convert list to tuple of tensors for comparison
        result = tuple(unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """unpack_sequence with keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed_sequences=packed)
        # Convert list to tuple of tensors for comparison
        result = tuple(unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """unpack_sequence round trip verification"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Verify round trip by checking each tensor matches
        matches = []
        for original, recovered in zip(sequences, unpacked):
            matches.append(torch.allclose(original, recovered))
        result = tuple(matches)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """unpack_sequence with unsorted sequences"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4])
        c = torch.tensor([5, 6])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Convert list to tuple of shapes for comparison
        result = tuple(t.shape for t in unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """unpack_sequence with float data"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        c = torch.tensor([6.0])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Convert list to tuple of tensors for comparison
        result = tuple(unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """unpack_sequence with multi-dimensional tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(15).float().reshape(3, 5)
        b = torch.arange(10).float().reshape(2, 5)
        c = torch.arange(5).float().reshape(1, 5)
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Convert list to tuple of shapes for comparison
        result = tuple(t.shape for t in unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """unpack_sequence preserves data types"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([4, 5], dtype=torch.float32)
        sequences = [a, b]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Convert list to tuple of tensors for comparison
        result = tuple(unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """unpack_sequence with single element sequences"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1])
        b = torch.tensor([2])
        c = torch.tensor([3])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Convert list to tuple of tensors for comparison
        result = tuple(unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        c = torch.tensor([6])
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed_sequences=packed)
        # Convert list to tuple of tensors for comparison
        result = tuple(unpacked)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0], requires_grad=True)
        c = torch.tensor([6.0], requires_grad=True)
        sequences = [a, b, c]
        packed = torch.nn.utils.rnn.pack_sequence(sequences)
        unpacked = torch.nn.utils.rnn.unpack_sequence(packed)
        # Compute gradient through unpacked sequences
        total = sum(u.sum() for u in unpacked)
        total.backward()
        # Return shapes of unpacked tensors and gradients
        result = tuple(t.shape for t in unpacked)
        a_grad = a.grad
        b_grad = b.grad
        c_grad = c.grad
    """
    )
    obj.run(
        pytorch_code,
        ["result", "a_grad", "b_grad", "c_grad"],
        check_stop_gradient=False,
    )
