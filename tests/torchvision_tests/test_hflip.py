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

obj = APIBase("torchvision.transforms.functional.hflip")
img_obj = ImageAPIBase("torchvision.transforms.functional.hflip")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]],
            [[9, 10],
             [11, 12]]
        ])
        result = F.hflip(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        result = F.hflip(torch.tensor([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (2, 2))
        img.putpixel((0, 0), (255, 0, 0))  # Red
        img.putpixel((1, 0), (0, 255, 0))  # Green
        img.putpixel((0, 1), (0, 0, 255))  # Blue
        img.putpixel((1, 1), (255, 255, 0))  # Yellow
        result = F.hflip(img=img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('L', (3, 3))
        img.putpixel((0, 0), 50)
        img.putpixel((1, 0), 100)
        img.putpixel((2, 0), 150)
        img.putpixel((0, 1), 200)
        img.putpixel((1, 1), 250)
        img.putpixel((2, 1), 100)
        img.putpixel((0, 2), 150)
        img.putpixel((1, 2), 200)
        img.putpixel((2, 2), 250)
        result = F.hflip(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])
