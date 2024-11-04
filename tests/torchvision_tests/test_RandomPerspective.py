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

obj = APIBase("torchvision.transforms.RandomPerspective")
img_obj = ImageAPIBase("torchvision.transforms.RandomPerspective")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomPerspective, InterpolationMode
        torch.manual_seed(0)
        transform = RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=InterpolationMode.BILINEAR, fill=0)
        img = torch.tensor([
            [[255, 0, 0],
             [0, 255, 0],
             [0, 0, 255]],
            [[255, 255, 0],
             [0, 255, 255],
             [255, 0, 255]],
            [[128, 128, 128],
             [64, 64, 64],
             [32, 32, 32]]
        ], dtype=torch.float)
        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomPerspective, InterpolationMode
        torch.manual_seed(1)
        mode = InterpolationMode.NEAREST
        transform = RandomPerspective(distortion_scale=0.7, p=0.0, interpolation=mode, fill=[1])
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
        ], dtype=torch.float)
        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        from torchvision.transforms import RandomPerspective, InterpolationMode
        import random
        random.seed(2)
        transform = RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=InterpolationMode.BICUBIC, fill=(255, 255, 255))
        img = Image.new('RGB', (4, 4), color=(255, 255, 255))
        img.putpixel((0, 0), (255, 0, 0))  # Red
        img.putpixel((3, 3), (0, 255, 0))  # Green
        img.putpixel((1, 1), (0, 0, 255))  # Blue
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        from torchvision.transforms import RandomPerspective, InterpolationMode
        import random
        random.seed(3)
        transform = RandomPerspective(distortion_scale=0.6, p=0.8, interpolation=InterpolationMode.NEAREST, fill=0)
        img = Image.new('L', (5, 5), color=128)  # Gray image
        img.putpixel((0, 0), 50)
        img.putpixel((4, 4), 200)
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomPerspective, InterpolationMode
        torch.manual_seed(4)
        transform = RandomPerspective(p=0.3, interpolation=InterpolationMode.BILINEAR, fill=[0, 0, 0], distortion_scale=0.4)
        img = torch.tensor([
            [
                [[10, 20, 30, 40, 50],
                 [60, 70, 80, 90, 100],
                 [110, 120, 130, 140, 150],
                 [160, 170, 180, 190, 200],
                 [210, 220, 230, 240, 250]],
                [[255, 0, 255, 0, 255],
                 [0, 255, 0, 255, 0],
                 [255, 0, 255, 0, 255],
                 [0, 255, 0, 255, 0],
                 [255, 0, 255, 0, 255]],
                [[128, 128, 128, 128, 128],
                 [64, 64, 64, 64, 64],
                 [32, 32, 32, 32, 32],
                 [16, 16, 16, 16, 16],
                 [8, 8, 8, 8, 8]]
            ],
            [
                [[5, 10, 15, 20, 25],
                 [30, 35, 40, 45, 50],
                 [55, 60, 65, 70, 75],
                 [80, 85, 90, 95, 100],
                 [105, 110, 115, 120, 125]],
                [[0, 255, 0, 255, 0],
                 [255, 0, 255, 0, 255],
                 [0, 255, 0, 255, 0],
                 [255, 0, 255, 0, 255],
                 [0, 255, 0, 255, 0]],
                [[8, 16, 24, 32, 40],
                 [16, 32, 48, 64, 80],
                 [24, 48, 72, 96, 120],
                 [32, 64, 96, 128, 160],
                 [40, 80, 120, 160, 200]]
            ]
        ], dtype=torch.float)
        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        from torchvision.transforms import RandomPerspective, InterpolationMode
        import random
        random.seed(5)
        transform = RandomPerspective(0.9, 0.9, InterpolationMode.BILINEAR, (255, 255, 255, 255))
        img = Image.new('RGBA', (6, 6), color=(0, 0, 255, 128))
        img.putpixel((0, 0), (255, 0, 0, 255))
        img.putpixel((5, 5), (0, 255, 0, 255))
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)
