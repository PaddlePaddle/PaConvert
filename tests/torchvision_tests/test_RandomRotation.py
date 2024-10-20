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

obj = APIBase("torchvision.transforms.RandomRotation")
img_obj = ImageAPIBase("torchvision.transforms.RandomRotation")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomRotation, InterpolationMode
        torch.manual_seed(0)
        degrees = 45
        rotation = RandomRotation(degrees=degrees)
        img = torch.tensor([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]],
            [[9, 10],
             [11, 12]]
        ], dtype=torch.float)
        result = rotation(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomRotation, InterpolationMode
        from PIL import Image
        import random
        random.seed(1)
        degrees = [-30, 30]
        rotation = RandomRotation(degrees=degrees, interpolation=InterpolationMode.BILINEAR, expand=True)
        img = Image.new('RGB', (3, 3), color=(255, 0, 0))
        img.putpixel((0, 0), (0, 255, 0))
        img.putpixel((2, 2), (0, 0, 255))
        result = rotation(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomRotation, InterpolationMode
        torch.manual_seed(2)
        degrees = 90
        center = (1, 1)
        fill = [255, 255, 255]
        rotation = RandomRotation(degrees=degrees, center=center, fill=fill)
        img = torch.tensor([
            [[10, 20, 30],
             [40, 50, 60],
             [70, 80, 90]],
            [[15, 25, 35],
             [45, 55, 65],
             [75, 85, 95]],
            [[12, 22, 32],
             [42, 52, 62],
             [72, 82, 92]]
        ], dtype=torch.float)
        result = rotation(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomRotation, InterpolationMode
        from PIL import Image
        import random
        random.seed(5)
        degrees = (-90, 90)
        center = (2, 2)
        fill = (0, 0, 255, 128)
        rotation = RandomRotation(degrees=degrees, interpolation=InterpolationMode.BICUBIC, expand=True, center=center, fill=fill)
        img = Image.new('RGBA', (5, 5), color=(0, 0, 255, 128))
        img.putpixel((0, 0), (255, 0, 0, 255))
        img.putpixel((4, 4), (0, 255, 0, 255))
        result = rotation(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)
