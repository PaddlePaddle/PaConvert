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

obj = APIBase("torch.nn.Module.named_modules")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        from collections import OrderedDict
        l = nn.Linear(2, 2)
        net = nn.Sequential(OrderedDict([
                        ('wfs', l),
                        ('wfs1', l),
                        ('wfs', l),
                        ('wfs1', l)]
                        ))
        z = net.named_modules(prefix="wfs", remove_duplicate=True)
        name_list = []
        for idx,m in enumerate(z):
            name_list.append(m[0])
        result = name_list
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        from collections import OrderedDict
        l = nn.Linear(2, 2)
        net = nn.Sequential(OrderedDict([
                        ('wfs', l),
                        ('wfs1', l)
                        ]))
        z = net.named_modules(prefix="wfs", remove_duplicate=False)
        name_list = []
        for idx,m in enumerate(z):
            name_list.append(m[0])
        result = name_list
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        from collections import OrderedDict
        l = nn.Linear(2, 2)
        net = nn.Sequential(OrderedDict([
                        ('wfs', l),
                        ('wfs1', l)
                        ]))
        memo = set()
        z = net.named_modules(prefix="wfs", memo=memo)
        name_list = []
        for idx,m in enumerate(z):
            name_list.append(m[0])
        result = name_list
        """
    )
    obj.run(pytorch_code, ["result"])
