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

obj = APIBase("torchvision.transforms.rotate")
img_obj = ImageAPIBase("torchvision.transforms.rotate")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        torch.manual_seed(0)
        img = torch.tensor([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]],
            [[9, 10],
             [11, 12]]
        ], dtype=torch.float)
        result = rotate(img=img, angle=45, interpolation=InterpolationMode.BILINEAR,
                       expand=True, center=(1,1), fill=[0, 0, 0])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        from PIL import Image
        img = Image.new('RGB', (3, 3), color=(255, 0, 0))
        img.putpixel((0, 0), (0, 255, 0))
        img.putpixel((2, 2), (0, 0, 255))
        result = rotate(img, 30, InterpolationMode.BILINEAR, True, (1,1), [128, 128, 128])
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        torch.manual_seed(2)
        img = torch.tensor([
            [[10, 20],
             [30, 40]],
            [[50, 60],
             [70, 80]]
        ], dtype=torch.float)
        result = rotate(fill=[255, 255], center=(0,0), expand=True,
                       interpolation=InterpolationMode.NEAREST, angle=90, img=img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        from PIL import Image
        img = Image.new('RGB', (4, 4), color=(255, 0, 0))
        result = rotate(img=img, angle=45)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        from PIL import Image
        img = Image.new('RGB', (3, 3), color=(255, 0, 0))
        result = rotate(img, angle=60, interpolation=InterpolationMode.NEAREST,
                       expand=True, center=(1,1))
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        from PIL import Image
        img = Image.new('RGBA', (5, 5), color=(0, 0, 255, 128))
        result = rotate(img, 90, center=(2,2), fill=(0, 0, 255, 128))
        """
    )
    img_obj.run(pytorch_code, ["result"])
