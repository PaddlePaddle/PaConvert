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

obj = APIBase("torch.nn.utils.vector_to_parameters")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        model = nn.Linear(10, 20)
        a = torch.nn.utils.parameters_to_vector(model.parameters())
        b = torch.nn.utils.vector_to_parameters(a, model.parameters())
        result = a.detach()
        """
    )
    obj.run(pytorch_code, ["result", "b"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        model = nn.Linear(10, 20)
        a = torch.nn.utils.parameters_to_vector(model.parameters())
        b = torch.nn.utils.vector_to_parameters(vec=a, parameters=model.parameters())
        result = a.detach()
        """
    )
    obj.run(pytorch_code, ["result", "b"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        model = nn.Linear(10, 20)
        a = torch.nn.utils.parameters_to_vector(model.parameters())
        b = torch.nn.utils.vector_to_parameters(parameters=model.parameters(), vec=a)
        result = a.detach()
        """
    )
    obj.run(pytorch_code, ["result", "b"], check_value=False)
