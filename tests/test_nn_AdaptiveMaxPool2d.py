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

obj = APIBase("torch.nn.AdaptiveMaxPool2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(5)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(output_size=5)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(5, False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


# The second return value's type of torch is int64 and the second return value's type of paddle is int32.
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(5, True)
        result, index = model(x)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(output_size=5, return_indices=False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])
