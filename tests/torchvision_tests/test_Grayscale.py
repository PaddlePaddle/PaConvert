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

obj = APIBase("torchvision.transforms.Grayscale")
img_obj = ImageAPIBase("torchvision.transforms.Grayscale")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms
        grayscale = transforms.Grayscale(num_output_channels=1)
        img = torch.tensor([
            [[0.2, 0.4],
             [0.6, 0.8]],
            [[0.1, 0.3],
             [0.5, 0.7]],
            [[0.0, 0.2],
             [0.4, 0.6]]
        ])
        result = grayscale(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms
        grayscale = transforms.Grayscale(3)
        img = torch.tensor([
            [[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6],
             [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7],
             [0.8, 0.9, 1.0]],
            [[0.3, 0.4, 0.5],
             [0.6, 0.7, 0.8],
             [0.9, 1.0, 1.1]]
        ])
        result = grayscale(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms as transforms
        num = 1
        grayscale = transforms.Grayscale(num_output_channels=num)
        img = Image.new('RGB', (2, 2), color=(255, 0, 0))  # Red image
        result = grayscale(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms as transforms
        grayscale = transforms.Grayscale(num_output_channels=3)
        img = Image.new('RGB', (3, 3), color=(0, 255, 0))  # Green image
        result = grayscale(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms
        grayscale = transforms.Grayscale(1)
        img = torch.tensor([
            [
                [[0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3]],
                [[0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.6]],
                [[0.9, 0.9, 0.9, 0.9],
                 [0.9, 0.9, 0.9, 0.9],
                 [0.9, 0.9, 0.9, 0.9],
                 [0.9, 0.9, 0.9, 0.9]]
            ]
        ])
        result = grayscale(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms as transforms
        grayscale = transforms.Grayscale(num_output_channels=3)
        img = Image.new('RGB', (5, 5), color=(0, 0, 255))  # Blue image
        result = grayscale(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])
