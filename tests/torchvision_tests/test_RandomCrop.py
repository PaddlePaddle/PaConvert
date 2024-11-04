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

obj = APIBase("torchvision.transforms.RandomCrop")
img_obj = ImageAPIBase("torchvision.transforms.RandomCrop")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomCrop
        torch.manual_seed(0)
        size = 2
        crop = RandomCrop(size=size)
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
        from torchvision.transforms import RandomCrop
        torch.manual_seed(1)
        size = [3, 3]
        padding = 1
        crop = RandomCrop(size=size, padding=padding, fill=0, padding_mode='constant')
        from PIL import Image
        img = Image.new('RGB', (4, 4), color=(255, 255, 255))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((3, 3), (0, 255, 0))
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomCrop
        torch.manual_seed(3)
        size = (3, 3)
        padding = [1, 1, 1, 1]
        pad_if_needed = True
        fill = (0, 0, 0)
        padding_mode = 'reflect'
        crop = RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)
        from PIL import Image
        img = Image.new('RGB', (3, 3), color=(100, 100, 100))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((2, 2), (0, 255, 0))
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomCrop
        torch.manual_seed(5)
        size = (4, 4)
        padding = 2
        pad_if_needed = True
        padding_mode = 'edge'
        fill = 0
        crop = RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)
        from PIL import Image
        img = Image.new('RGB', (6, 6), color=(50, 100, 150))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((5, 5), (0, 255, 0))
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomCrop
        torch.manual_seed(5)
        size = (4, 4)
        padding = 2
        pad_if_needed = True
        padding_mode = 'edge'
        fill = 0
        crop = RandomCrop(size, padding, pad_if_needed, fill, padding_mode)
        from PIL import Image
        img = Image.new('RGB', (6, 6), color=(50, 100, 150))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((5, 5), (0, 255, 0))
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomCrop
        torch.manual_seed(5)
        size = (4, 4)
        padding = 2
        pad_if_needed = True
        padding_mode = 'edge'
        fill = 0
        crop = RandomCrop(pad_if_needed=pad_if_needed, size=size, padding=padding, fill=fill, padding_mode=padding_mode)
        from PIL import Image
        img = Image.new('RGB', (6, 6), color=(50, 100, 150))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((5, 5), (0, 255, 0))
        result = crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)
