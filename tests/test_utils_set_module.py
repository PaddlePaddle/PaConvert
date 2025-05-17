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


class SetModuleBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert pytorch_result == paddle_result


obj = SetModuleBase("torch.utils.set_module")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        class CustomLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(3, 3))
        layer = CustomLayer()
        torch.utils.set_module(layer, "MyCustomLayer")
        result = layer.__module__
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        class CustomLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(3, 3))
        layer = CustomLayer()
        torch.utils.set_module(obj=layer, mod="MyCustomLayer")
        result = layer.__module__
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        class CustomLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(3, 3))
        layer = CustomLayer()
        torch.utils.set_module(mod="MyCustomLayer", obj=layer)
        result = layer.__module__
        """
    )
    obj.run(pytorch_code, ["result"])
