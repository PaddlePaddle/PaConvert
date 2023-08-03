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

obj = APIBase("torch.Tensor.get_device")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = None
        if torch.cuda.is_available():
            x = torch.tensor([[1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0]]).cuda()
            result = x.get_device()

        """
    )
    obj.run(pytorch_code, ["result"])


# for CPU Tensor, paddle.Tensor.place.gpu_device_id return 0, while torch return -1
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = None
        x = torch.tensor([[1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0]])
        result = x.get_device()
        """
    )
    obj.run(pytorch_code, ["result"])
