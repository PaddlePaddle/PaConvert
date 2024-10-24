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
from torchvision_tests.image_apibase import ImageAPIBase

obj = APIBase("torchvision.transforms.functional.crop")
img_obj = ImageAPIBase("torchvision.transforms.functional.crop")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]],
            [[17, 18, 19, 20],
             [21, 22, 23, 24],
             [25, 26, 27, 28],
             [29, 30, 31, 32]],
            [[33, 34, 35, 36],
             [37, 38, 39, 40],
             [41, 42, 43, 44],
             [45, 46, 47, 48]]
        ])
        top, left, height, width = 1, 1, 2, 2
        result = F.crop(img, top, left, height, width)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (4, 4), color=(255, 0, 0))
        result = F.crop(img, 0, 0, 2, 2)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        top, left, height, width = 1, 1, 3, 3
        result = F.crop(Image.new('RGB', (3, 3), color=(0, 255, 0)), top=top, left=left, height=height, width=width)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [[10, 20, 30, 40, 50],
             [60, 70, 80, 90, 100],
             [110, 120, 130, 140, 150],
             [160, 170, 180, 190, 200],
             [210, 220, 230, 240, 250]],
            [[255, 245, 235, 225, 215],
             [205, 195, 185, 175, 165],
             [155, 145, 135, 125, 115],
             [105, 95, 85, 75, 65],
             [55, 45, 35, 25, 15]],
            [[5, 15, 25, 35, 45],
             [55, 65, 75, 85, 95],
             [105, 115, 125, 135, 145],
             [155, 165, 175, 185, 195],
             [205, 215, 225, 235, 245]]
        ])
        top, left, height, width = 2, 2, 2, 2
        result = F.crop(img, top, left, height, width)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (5, 5), color=(0, 0, 255))
        top, left, height, width = 3, 3, 2, 2
        result = F.crop(img, top=top, height=height, left=left, width=width)
        """
    )
    img_obj.run(pytorch_code, ["result"])
