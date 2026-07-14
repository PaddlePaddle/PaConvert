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

obj = APIBase("torch.nn.modules.rnn.LSTM")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 2)
        result, _ = model(torch.randn(5, 3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(input_size=10, hidden_size=20, num_layers=2)
        result, _ = model(torch.randn(5, 3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 2, bias=True, batch_first=False)
        result, _ = model(torch.randn(5, 3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(input_size=10, hidden_size=20, num_layers=2, bias=True, batch_first=True)
        result, _ = model(torch.randn(3, 5, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 1, batch_first=True)
        result, _ = model(torch.randn(3, 5, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 2, dropout=0.2)
        result, _ = model(torch.randn(5, 3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 2, bidirectional=True)
        h0 = torch.randn(4, 3, 20)
        c0 = torch.randn(4, 3, 20)
        result, _ = model(torch.randn(5, 3, 10), (h0, c0))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 2, bias=False)
        result, _ = model(torch.randn(5, 3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(5, 10, 1)
        result, _ = model(torch.randn(3, 2, 5))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.LSTM(10, 20, 3, dropout=0.3, batch_first=True)
        result, _ = model(torch.randn(3, 5, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
