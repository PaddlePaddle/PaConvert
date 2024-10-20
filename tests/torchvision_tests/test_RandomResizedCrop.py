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

obj = APIBase("torchvision.transforms.RandomResizedCrop")
img_obj = ImageAPIBase("torchvision.transforms.RandomResizedCrop")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomResizedCrop
        torch.manual_seed(0)
        size = 2
        crop = RandomResizedCrop(size=size)
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
        result = crop(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomResizedCrop, InterpolationMode
        torch.manual_seed(1)
        size = (3, 3)
        scale = (0.5, 0.9)
        ratio = (0.8, 1.2)
        crop = RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BILINEAR)
        from PIL import Image
        img = Image.new('RGB', (4, 4), color=(255, 255, 255))  # White image
        img.putpixel((0, 0), (255, 0, 0))  # Red
        img.putpixel((3, 3), (0, 255, 0))  # Green
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomResizedCrop, InterpolationMode
        torch.manual_seed(1)
        size = [3, 3]
        scale = (0.5, 0.9)
        ratio = (0.8, 1.2)
        crop = RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BILINEAR)
        from PIL import Image
        img = Image.new('RGB', (4, 4), color=(255, 255, 255))  # White image
        img.putpixel((0, 0), (255, 0, 0))  # Red
        img.putpixel((3, 3), (0, 255, 0))  # Green
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomResizedCrop, InterpolationMode
        torch.manual_seed(3)
        size = (3, 3)
        scale = (0.8, 0.8)  # Fixed scale
        ratio = (1.0, 1.0)  # Fixed aspect ratio
        crop = RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BICUBIC)
        from PIL import Image
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
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomResizedCrop, InterpolationMode
        torch.manual_seed(5)
        size = (4, 4)
        scale = (0.3, 0.8)
        ratio = (0.75, 1.3333)
        crop = RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BICUBIC)
        from PIL import Image
        img = Image.new('RGBA', (6, 6), color=(0, 0, 255, 128))  # Semi-transparent Blue
        img.putpixel((0, 0), (255, 0, 0, 255))  # Red
        img.putpixel((5, 5), (0, 255, 0, 255))  # Green
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)
