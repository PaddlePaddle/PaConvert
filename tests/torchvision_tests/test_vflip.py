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

obj = APIBase("torchvision.transforms.functional.vflip")
img_obj = ImageAPIBase("torchvision.transforms.functional.vflip")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (2, 2), color=(255, 0, 0))  # Red image
        result = F.vflip(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        result = F.vflip(Image.new('L', (4, 4), color=128))
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]
            ],
            [
                [19, 20, 21],
                [22, 23, 24],
                [25, 26, 27]
            ]
        ], dtype=torch.float)
        result = F.vflip(img=img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]
                ],
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32]
                ],
                [
                    [33, 34, 35, 36],
                    [37, 38, 39, 40],
                    [41, 42, 43, 44],
                    [45, 46, 47, 48]
                ]
            ],
            [
                [
                    [49, 50, 51, 52],
                    [53, 54, 55, 56],
                    [57, 58, 59, 60],
                    [61, 62, 63, 64]
                ],
                [
                    [65, 66, 67, 68],
                    [69, 70, 71, 72],
                    [73, 74, 75, 76],
                    [77, 78, 79, 80]
                ],
                [
                    [81, 82, 83, 84],
                    [85, 86, 87, 88],
                    [89, 90, 91, 92],
                    [93, 94, 95, 96]
                ]
            ]
        ], dtype=torch.float)
        result = F.vflip(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGBA', (5, 5), color=(0, 0, 255, 128))
        result = F.vflip(img=img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        result = F.vflip(torch.tensor([
            [
                [
                    [1, 2],
                    [3, 4]
                ]
            ]
        ], dtype=torch.float))
        """
    )
    obj.run(pytorch_code, ["result"])
