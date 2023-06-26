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

obj = APIBase("torch.nn.Module.named_children")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        from collections import OrderedDict
        l = nn.Linear(2, 2,bias=False)
        l1 = nn.Linear(2, 2,bias=False)
        model = nn.Sequential(OrderedDict([
                        ('wfs', l),
                        ('wfs1', l1)
                        ]))
        result = torch.Tensor([0,0])
        for name, module in model.named_children():
            result = module(result)
        """
    )
    obj.run(pytorch_code, ["result"])
