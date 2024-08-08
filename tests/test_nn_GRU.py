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

obj = APIBase("torch.nn.GRU")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)

            def forward(self, x):
                output, h_n = self.gru(x)
                return output, h_n

        x = torch.randn(5, 3, 10)  # (batch_size, seq_len, input_size)
        model = SimpleRNNModel()
        output, h_n = model(x)
        """
    )
    obj.run(pytorch_code, ["output", "h_n"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.gru = nn.GRU(10,5)

            def forward(self, x):
                output, h_n = self.gru(x)
                return output, h_n

        x = torch.randn(3, 10, 10)
        model = SimpleRNNModel()
        output, h_n = model(x)
        """
    )
    obj.run(pytorch_code, ["output", "h_n"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.gru = nn.GRU(batch_first=True, num_layers=2, bidirectional=True, hidden_size=20, input_size=10, dropout=0.5)

            def forward(self, x):
                output, h_n = self.gru(x)
                return output, h_n

        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        output, h_n = model(x)
        """
    )
    obj.run(pytorch_code, ["output", "h_n"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.gru = nn.GRU(10, 5, 2, bidirectional=False)

            def forward(self, x):
                output, h_n = self.gru(x)
                return output, h_n

        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        output, h_n = model(x)
        """
    )
    obj.run(pytorch_code, ["output", "h_n"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.gru = nn.GRU(10, 3, 2, True, False, 0.7, False)

            def forward(self, x):
                output, h_n = self.gru(x)
                return output, h_n


        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        output, h_n = model(x)
        """
    )
    obj.run(pytorch_code, ["output", "h_n"], check_value=False)
