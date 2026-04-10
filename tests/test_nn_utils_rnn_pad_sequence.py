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

obj = APIBase("torch.nn.utils.rnn.pad_sequence")


def test_case_1():
    """basic usage with default parameters"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        c = torch.tensor([6.0])
        result = pad_sequence([a, b, c])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """batch_first=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        c = torch.tensor([6.0])
        result = pad_sequence([a, b, c], batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """custom padding_value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = pad_sequence([a, b], padding_value=-1.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """all positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = pad_sequence([a, b], True, -1.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """all keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = pad_sequence(sequences=[a, b], batch_first=True, padding_value=0.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """keyword arguments in shuffled order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = pad_sequence(padding_value=2.0, batch_first=False, sequences=[a, b])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """2D input tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = torch.tensor([[7.0, 8.0]])
        result = pad_sequence([a, b], batch_first=True, padding_value=0.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """using torch.nn.utils.rnn.pad_sequence full path"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        result = torch.nn.utils.rnn.pad_sequence([a, b], batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """sequences of same length"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = pad_sequence([a, b], batch_first=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """integer dtype input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1, 2, 3, 4])
        b = torch.tensor([5, 6])
        result = pad_sequence([a, b], batch_first=False, padding_value=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """batch_first=False with custom padding_value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0])
        c = torch.tensor([7.0, 8.0, 9.0])
        result = pad_sequence([a, b, c], batch_first=False, padding_value=-100.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """single sequence input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        a = torch.tensor([1.0, 2.0, 3.0])
        result = pad_sequence([a])
        """
    )
    obj.run(pytorch_code, ["result"])
