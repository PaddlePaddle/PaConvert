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

obj = APIBase("torch.nn.utils.rnn.pack_padded_sequence")


def test_case_1():
    """Basic pack_padded_sequence with positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.randn(5, 3, 10)
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    """pack_padded_sequence with keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.randn(5, 3, 10)
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=lengths)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    """pack_padded_sequence with batch_first=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.randn(3, 5, 10)  # B x T x *
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    """pack_padded_sequence with enforce_sorted=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
        lengths = [2, 1, 3]
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        result = (packed.data, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """pack_padded_sequence with list of lengths"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.randn(5, 3, 10)
        lengths = [5, 3, 2]
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    """pack_padded_sequence sorted sequences"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[1, 1, 1], [2, 2, 0], [3, 0, 0]]).float()
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """pack_padded_sequence with trailing dimensions"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.randn(4, 2, 3, 5)  # T x B x * (3D trailing)
        lengths = torch.tensor([4, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    """pack_padded_sequence mixed arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.randn(5, 3, 10)
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=False)
        result = (packed.data, packed.batch_sizes)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
