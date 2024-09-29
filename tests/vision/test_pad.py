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
from vision.image_apibase import ImageAPIBase

obj = APIBase("torchvision.transforms.functional.pad")
img_obj = ImageAPIBase("torchvision.transforms.functional.pad")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F

        padding = 2
        fill = 0
        padding_mode = 'constant'

        img = torch.tensor([
            [[1, 2],
             [3, 4]],

            [[5, 6],
             [7, 8]],

            [[9, 10],
             [11, 12]]
        ], dtype=torch.float)

        result = F.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F

        padding = [1, 2, 3, 4]
        fill = 1.0
        padding_mode = 'constant'

        img = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],

            [[10, 11, 12],
             [13, 14, 15],
             [16, 17, 18]],

            [[19, 20, 21],
             [22, 23, 24],
             [25, 26, 27]]
        ], dtype=torch.float)

        result = F.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F

        padding = [2, 3]
        fill = (255, 0, 0)
        padding_mode = 'constant'

        img = Image.new('RGB', (2, 2), color=(0, 255, 0))

        result = F.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F

        padding = 1
        padding_mode = 'reflect'

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

        result = F.pad(img, padding=padding, padding_mode=padding_mode)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F

        padding = [1, 1, 1, 1]
        fill = (0, 0, 255, 128)
        padding_mode = 'symmetric'

        img = Image.new('RGBA', (5, 5), color=(0, 0, 255, 128))
        img.putpixel((0, 0), (255, 0, 0, 255))
        img.putpixel((4, 4), (0, 255, 0, 255))

        result = F.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)
        """
    )
    img_obj.run(pytorch_code, ["result"])
