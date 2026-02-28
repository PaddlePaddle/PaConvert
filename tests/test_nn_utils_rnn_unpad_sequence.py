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

obj = APIBase("torch.nn.utils.rnn.unpad_sequence")


def test_case_1():
    """basic usage with batch_first=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        c = torch.tensor([6.0])
        padded = pad_sequence([a, b, c], batch_first=True)
        lengths = torch.tensor([3, 2, 1])
        result = unpad_sequence(padded, lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """batch_first=False (default)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        padded = pad_sequence([a, b], batch_first=False)
        lengths = torch.tensor([3, 2])
        result = unpad_sequence(padded, lengths, batch_first=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """all positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        padded = pad_sequence([a, b], True)
        lengths = torch.tensor([3, 2])
        result = unpad_sequence(padded, lengths, True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """all keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        padded = pad_sequence([a, b], batch_first=True)
        lengths = torch.tensor([3, 2])
        result = unpad_sequence(padded_sequences=padded, lengths=lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """keyword arguments in shuffled order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        padded = pad_sequence([a, b], batch_first=True)
        lengths = torch.tensor([3, 2])
        result = unpad_sequence(batch_first=True, lengths=lengths, padded_sequences=padded)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """default batch_first (omitted)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        padded = pad_sequence([a, b])
        lengths = torch.tensor([3, 2])
        result = unpad_sequence(padded, lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """2D input tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = torch.tensor([[7.0, 8.0]])
        padded = pad_sequence([a, b], batch_first=True)
        lengths = torch.tensor([3, 1])
        result = unpad_sequence(padded, lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """using full path torch.nn.utils.rnn.unpad_sequence"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        padded = pad_sequence([a, b], batch_first=True)
        lengths = torch.tensor([3, 2])
        result = torch.nn.utils.rnn.unpad_sequence(padded, lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """single sequence"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        padded = pad_sequence([a], batch_first=True)
        lengths = torch.tensor([3])
        result = unpad_sequence(padded, lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """integer dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1, 2, 3, 4])
        b = torch.tensor([5, 6])
        padded = pad_sequence([a, b], batch_first=True)
        lengths = torch.tensor([4, 2])
        result = unpad_sequence(padded, lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """multiple sequences with varying lengths"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0])
        c = torch.tensor([7.0, 8.0, 9.0])
        padded = pad_sequence([a, b, c], batch_first=True)
        lengths = torch.tensor([4, 2, 3])
        result = unpad_sequence(padded, lengths, batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])
