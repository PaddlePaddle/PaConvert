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

obj = APIBase("torchvision.transforms.functional.perspective")
img_obj = ImageAPIBase(
    "torchvision.transforms.functional.perspective"
)  # Supports both Tensor and PIL Image


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import perspective

        startpoints = [[0, 0], [4, 0], [4, 4], [0, 4]]  # Original corners of a 5x5 image
        endpoints = [[0, 0], [4, 0], [3, 4], [0, 4]]  # Vertical skew

        from PIL import Image
        img = Image.new('RGB', (5, 5), color=(255, 255, 255))  # White image
        img.putpixel((0, 0), (255, 0, 0))  # Red
        img.putpixel((4, 4), (0, 255, 0))  # Green
        img.putpixel((2, 2), (0, 0, 255))  # Blue

        fill = 0, 0, 0
        result = perspective(img, startpoints, endpoints, InterpolationMode.BILINEAR, fill)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import perspective

        startpoints = [[0, 0], [4, 0], [4, 4], [0, 4]]  # Original corners of a 5x5 image
        endpoints = [[0, 0], [4, 0], [3, 4], [0, 4]]  # Vertical skew

        from PIL import Image
        img = Image.new('RGB', (5, 5), color=(255, 255, 255))  # White image
        img.putpixel((0, 0), (255, 0, 0))  # Red
        img.putpixel((4, 4), (0, 255, 0))  # Green
        img.putpixel((2, 2), (0, 0, 255))  # Blue

        result = perspective(img, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0))
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import perspective

        startpoints = [[0, 0], [3, 0], [3, 3], [0, 3]]
        endpoints = [[0, 0], [3, 0], [2.5, 3], [0, 3]]

        img = torch.tensor([
            [
                [[255, 0, 0, 0],
                 [0, 255, 0, 0],
                 [0, 0, 255, 0],
                 [0, 0, 0, 255]],

                [[255, 255, 0, 0],
                 [0, 255, 255, 0],
                 [255, 0, 255, 0],
                 [0, 255, 0, 255]],

                [[128, 128, 128, 128],
                 [64, 64, 64, 64],
                 [32, 32, 32, 32],
                 [16, 16, 16, 16]]
            ],
            [
                [[0, 0, 255, 255],
                 [0, 255, 0, 255],
                 [255, 0, 0, 255],
                 [255, 255, 255, 255]],

                [[0, 255, 255, 255],
                 [255, 255, 0, 255],
                 [0, 255, 255, 255],
                 [255, 255, 0, 255]],

                [[16, 32, 64, 128],
                 [32, 64, 128, 256],
                 [64, 128, 256, 512],
                 [128, 256, 512, 1024]]
            ]
        ], dtype=torch.float)

        result = perspective(img, startpoints, endpoints, interpolation=InterpolationMode.NEAREST, fill=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import perspective
        from PIL import Image

        startpoints = [[0, 0], [2, 0], [2, 2], [0, 2]]
        endpoints = [[0, 0], [2, 0], [2, 1.8], [0, 2]]

        img = Image.new('L', (3, 3), color=128)  # Gray image
        img.putpixel((0, 0), 50)
        img.putpixel((2, 2), 200)

        result = perspective(img, startpoints, endpoints, interpolation=InterpolationMode.NEAREST, fill=0)
        """
    )
    img_obj.run(pytorch_code, ["result"])
