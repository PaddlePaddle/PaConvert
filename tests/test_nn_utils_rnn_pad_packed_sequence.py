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

obj = APIBase("torch.nn.utils.rnn.pad_packed_sequence")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        # Compare output tensor and lengths
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-1.0)
        # Compare output tensor and lengths
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(padding_value=0.0, sequence=packed, batch_first=True)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """batch_first=False test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 1, 3], [5, 2, 0], [6, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=False)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=False)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """total_length parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=5)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Default arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 1, 3], [5, 2, 0], [6, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """All keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(input=seq, lengths=lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(sequence=packed, batch_first=True, padding_value=1.5, total_length=None)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D input with feature dimension"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(42)
        seq = torch.from_numpy(np.random.randn(5, 3, 10).astype(np.float32))
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=False)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=False)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Mixed positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, True, padding_value=-1.0)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """enforce_sorted=False test - check value=False due to reordering"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([2, 3, 1])  # unsorted
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(42)
        seq = torch.from_numpy(np.random.randn(5, 3, 10).astype(np.float32))
        seq.requires_grad = True
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=False)
        padded, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=False)
        loss = padded.sum()
        loss.backward()
        seq_grad = seq.grad
        """
    )
    obj.run(pytorch_code, ["loss", "seq_grad"], check_stop_gradient=False)


def test_case_12():
    """Expression as argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=3 + 2)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """Different data types - float64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """Different data types - int64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        seq = torch.tensor([[4, 5, 6], [1, 2, 0], [3, 0, 0]], dtype=torch.int64)
        lengths = torch.tensor([3, 2, 1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True)
        result, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        result = (result, lengths_out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """Round trip test - pack then pad should recover original"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(100)
        seq = torch.from_numpy(np.random.randn(5, 3, 10).astype(np.float32))
        lengths = torch.tensor([5, 3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=False)
        recovered, lengths_out = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=False)
        # Check if recovered matches original
        match = torch.allclose(seq, recovered)
        result = (recovered, lengths_out, match)
        """
    )
    obj.run(pytorch_code, ["result"])
