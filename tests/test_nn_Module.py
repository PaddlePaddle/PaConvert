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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.nn.Module")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()

                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

                with torch.no_grad():
                    self.fc1.weight.fill_(1.0)
                    self.fc1.bias.fill_(0.1)
                    self.fc2.weight.fill_(1.0)
                    self.fc2.bias.fill_(0.1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        model = MLP(784, 256, 10)

        pytorch_module = MLP(784, 256, 10)

        inputs = torch.ones([64, 784])
        result = model(inputs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()

                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        model = MLP(784, 256, 10)
        result = model.__class__.__name__
        """
    )
    obj.run(pytorch_code, ["result"])
