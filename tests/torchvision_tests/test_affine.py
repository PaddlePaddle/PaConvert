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

obj = APIBase("torchvision.transforms.functional.affine")
img_obj = ImageAPIBase("torchvision.transforms.affine")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms.functional import affine
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        result = affine(img, 30, (1, 1), 1.0, 0, InterpolationMode.NEAREST)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms.functional import affine
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(0, 255, 0))
        result = affine(img=img, angle=0, translate=(1, 1), scale=1.0, shear=0, interpolation=InterpolationMode.NEAREST)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms.functional import affine
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(0, 0, 255))
        result = affine(img=img, translate=(0, 0), angle=0, scale=1.5, shear=0, interpolation=InterpolationMode.NEAREST)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms.functional import affine
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(255, 255, 0))
        result = affine(img=img, angle=0, translate=(0, 0), scale=1.0, shear=10, interpolation=InterpolationMode.NEAREST, fill=0)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms.functional import affine
        from torchvision.transforms import InterpolationMode
        from PIL import Image

        img = Image.new('RGB', (10, 10), color=(0, 255, 255))
        result = affine(img=img, angle=30, translate=(0, 0), scale=1.0, shear=0, interpolation=InterpolationMode.NEAREST, center=(2, 2))
        """
    )
    img_obj.run(pytorch_code, ["result"])
