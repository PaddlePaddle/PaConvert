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

obj = APIBase("torch.nn.ParameterDict")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            f"param_{i}": nn.Parameter(torch.ones(i + 1, i + 1)) for i in range(10)
        })
        result = list(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict()
        result = list(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        param_dict = nn.ParameterDict({
            'param1': nn.Parameter(torch.randn(3, 3)),
            'param2': nn.Parameter(torch.ones(2, 2)),
        })
        result = list(param_dict)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            f"param_{i}": nn.Parameter(torch.ones(i + 1, i + 1)) for i in range(3)
        })
        choices.update({
            f"new_param_{i}": nn.Parameter(torch.zeros(i + 2, i + 2)) for i in range(3, 5)
        })
        result = list(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            f"param_{i}": nn.Parameter(torch.ones(i + 1, i + 1)) for i in range(3)
        })
        choices.update({
            f"new_param_{i}": nn.Parameter(torch.zeros(i + 2, i + 2)) for i in range(3, 5)
        })
        result = list(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict(parameters={
            'a': nn.Parameter(torch.ones(2, 3)),
            'b': nn.Parameter(torch.zeros(4)),
        })
        result = list(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict([
            ('a', nn.Parameter(torch.ones(2, 3))),
            ('b', nn.Parameter(torch.zeros(4))),
        ])
        result = list(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({'w': nn.Parameter(torch.ones(2, 3))})
        result = choices['w']
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            'a': nn.Parameter(torch.ones(1)),
            'b': nn.Parameter(torch.ones(2)),
            'c': nn.Parameter(torch.ones(3)),
        })
        result = len(choices)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            'a': nn.Parameter(torch.ones(1)),
            'b': nn.Parameter(torch.ones(2)),
        })
        result = list(choices.keys())
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            'a': nn.Parameter(torch.ones(2, 3)),
            'b': nn.Parameter(torch.zeros(4)),
        })
        result = list(choices.values())
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({'a': nn.Parameter(torch.ones(2))})
        result = 'a' in choices
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict(parameters={
            'a': nn.Parameter(torch.ones(2, 3)),
            'b': nn.Parameter(torch.zeros(4)),
        })
        result = choices['a']
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict(parameters={
            'a': nn.Parameter(torch.ones(2, 3)),
            'b': nn.Parameter(torch.zeros(4)),
        })
        result = choices['b']
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        choices = nn.ParameterDict({
            'a': nn.Parameter(torch.ones(2, 3)),
            'b': nn.Parameter(torch.zeros(4)),
        })
        result = choices.pop('a')
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        pd1 = nn.ParameterDict({
            'a': nn.Parameter(torch.ones(2, 3)),
            'b': nn.Parameter(torch.zeros(4)),
        })
        pd2 = nn.ParameterDict(pd1)
        result = list(pd2)
        """
    )
    obj.run(pytorch_code, ["result"])
