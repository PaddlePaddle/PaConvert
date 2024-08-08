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

import numpy as np
from apibase import APIBase


class nn_DataParallelAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        (
            pytorch_numpy,
            paddle_numpy,
        ) = pytorch_result.cpu().detach().numpy(), paddle_result.numpy(False)
        assert (
            pytorch_numpy.dtype == paddle_numpy.dtype
        ), "API ({}): dtype mismatch, torch dtype is {}, paddle dtype is {}".format(
            name, pytorch_numpy.dtype, paddle_numpy.dtype
        )
        if check_value:
            assert np.allclose(
                pytorch_numpy, paddle_numpy, rtol=rtol, atol=atol
            ), "API ({}): paddle result has diff with pytorch result".format(name)


obj = nn_DataParallelAPIBase("torch.nn.DaraParallel")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(3, 2)
                nn.init.constant_(self.fc.weight, 1.0)
                nn.init.constant_(self.fc.bias, 0.0)

            def forward(self, x):
                return self.fc(x)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleModel().to(device)
        model = nn.DataParallel(model)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(3, 2)
                nn.init.constant_(self.fc.weight, 1.0)
                nn.init.constant_(self.fc.bias, 0.0)

            def forward(self, x):
                return self.fc(x)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = nn.DataParallel(SimpleModel(), device_ids=[0])
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(3, 2)
                nn.init.constant_(self.fc.weight, 1.0)
                nn.init.constant_(self.fc.bias, 0.0)

            def forward(self, x):
                return self.fc(x)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = nn.DataParallel(SimpleModel(), device_ids=[0], output_device=0)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(3, 2)
                nn.init.constant_(self.fc.weight, 1.0)
                nn.init.constant_(self.fc.bias, 0.0)

            def forward(self, x):
                return self.fc(x)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = nn.DataParallel(SimpleModel(), dim=1)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], unsupport="True", reason="unsupported args dim")


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(3, 2, bias=False)
                nn.init.constant_(self.fc.weight, 1.0)

            def forward(self, x):
                return self.fc(x)

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleModel().to(device)
        model = nn.DataParallel(model)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])
