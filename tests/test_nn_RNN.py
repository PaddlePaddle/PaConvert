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

obj = APIBase("torch.nn.RNN")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, nonlinearity='tanh',bias=True,batch_first=True, dropout=0.5,
                                    bidirectional=True,device=None, dtype=None)

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(10, 20)

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(3, 5, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(batch_first=True, num_layers=2, hidden_size=20, input_size=10, dropout=0.5)

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(input_size=10, hidden_size=20, nonlinearity='relu')

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(3, 5, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(10, 20, 3,'relu')

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(3, 5, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, nonlinearity='tanh', bias=False, batch_first=True, dropout=0.5,
                                    bidirectional=True)

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle not support bias =False",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(10, 20, 3, 'relu', True, False, 0.6, bidirectional=True)

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(3, 5, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleRNNModel(nn.Module):
            def __init__(self):
                super(SimpleRNNModel, self).__init__()
                self.rnn = nn.RNN(10, 20, 2,  batch_first=True, dropout=0.5, bias=True, nonlinearity='relu',
                                    bidirectional=True)

            def forward(self, x):
                output, hidden = self.rnn(x)
                return output

        x = torch.randn(5, 3, 10)
        model = SimpleRNNModel()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
