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

obj = APIBase("torchvision.transforms.RandomAffine")
img_obj = ImageAPIBase("torchvision.transforms.RandomAffine")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomAffine
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        transform = RandomAffine(degrees=30)
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomAffine
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(0, 255, 0))
        transform = RandomAffine(degrees=0, translate=(0.1, 0.1))
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomAffine
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(0, 0, 255))
        transform = RandomAffine(degrees=0, scale=(0.5, 1.5))
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomAffine
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(255, 255, 0))
        transform = RandomAffine(degrees=0, shear=10)
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomAffine, InterpolationMode
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(255, 0, 255))
        transform = RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=20, interpolation=InterpolationMode.BILINEAR, fill=255)
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomAffine
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(0, 255, 255))
        transform = RandomAffine(degrees=(-30, 30), center=(2, 2))
        result = transform(img)
        """
    )
    img_obj.run(pytorch_code, ["result"], check_value=False)
