# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torchvision.ops.roi_align", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import roi_align
        input = torch.tensor([[[[1.0] * 32] * 32] * 245] * 2, dtype=torch.float)
        boxes = torch.tensor([[0, 4, 4, 7, 7], [1, 5, 5, 10, 10]], dtype=torch.float)
        result = roi_align(input=input, boxes=boxes, output_size=7, spatial_scale=1.0, sampling_ratio=-1, aligned=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import roi_align
        input = torch.tensor([[[[1.0] * 32] * 32] * 245], dtype=torch.float)
        boxes = torch.tensor([[0, 1, 1, 10, 10]], dtype=torch.float)
        result = roi_align(input, boxes, (7, 7), 1.0, -1, False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import roi_align
        input = torch.tensor([[[[1.0] * 32] * 32] * 245] * 2, dtype=torch.float)
        boxes = [torch.tensor([[0, 0, 10, 10]], dtype=torch.float)]
        result = roi_align(aligned=False, sampling_ratio=-1, spatial_scale=1.0, boxes=boxes, input=input, output_size=(7, 7))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import roi_align
        input = torch.tensor([[[[1.0] * 32] * 32] * 245] * 2, dtype=torch.float)
        boxes = [
            torch.tensor([[0, 0, 10, 10], [1, 1, 15, 15]], dtype=torch.float),
            torch.tensor([[0, 0, 8, 8]], dtype=torch.float)
        ]
        result = roi_align(input, boxes, output_size=(7, 7), spatial_scale=1.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import roi_align
        input = torch.tensor([[[[1.0] * 32] * 32] * 245] * 2, dtype=torch.float)
        boxes = torch.tensor([[0, 0, 0, 10, 10]], dtype=torch.float)
        result = roi_align(input, boxes, (7, 7))
        """
    )
    obj.run(pytorch_code, ["result"])
