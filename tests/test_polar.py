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

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.polar')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy as np
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        result = torch.polar(abs, angle)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy as np
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        out = torch.tensor([1, 2], dtype=torch.complex128)
        result = torch.polar(abs, angle, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy as np
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        result = torch.polar(torch.tensor([1, 2], dtype=torch.float64), angle)
        '''
    )
    obj.run(pytorch_code, ['result'])
