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

obj = APIBase("torchvision.transforms.Resize")
img_obj = ImageAPIBase("torchvision.transforms.Resize")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
        from PIL import Image
        torch.manual_seed(1)
        size = (3, 3)
        img = Image.new('RGB', (4, 4), color=(255, 255, 255))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((3, 3), (0, 255, 0))
        result = resize(img=img, size=size, interpolation=InterpolationMode.BILINEAR)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
        from PIL import Image
        torch.manual_seed(3)
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
        result = resize(img, size=3, interpolation=InterpolationMode.BICUBIC)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
        torch.manual_seed(4)
        size = [4, 4]
        img = torch.tensor([
            [
                [[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25]],
                [[26, 27, 28, 29, 30],
                 [31, 32, 33, 34, 35],
                 [36, 37, 38, 39, 40],
                 [41, 42, 43, 44, 45],
                 [46, 47, 48, 49, 50]],
                [[51, 52, 53, 54, 55],
                 [56, 57, 58, 59, 60],
                 [61, 62, 63, 64, 65],
                 [66, 67, 68, 69, 70],
                 [71, 72, 73, 74, 75]]
            ]
        ], dtype=torch.float)
        img = img[0]
        result = resize(img=img, size=size, interpolation=InterpolationMode.NEAREST)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
        from PIL import Image
        torch.manual_seed(5)
        size = (4, 4)
        img = Image.new('RGBA', (6, 6), color=(0, 0, 255, 128))
        img.putpixel((0, 0), (255, 0, 0, 255))
        img.putpixel((5, 5), (0, 255, 0, 255))
        result = resize(img, size, InterpolationMode.BICUBIC)
        """
    )
    img_obj.run(pytorch_code, ["result"])
