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

obj = APIBase("torchvision.transforms.Compose")
img_obj = ImageAPIBase("torchvision.transforms.Compose")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms

        composed = transforms.Compose([
            transforms.CenterCrop(2),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img = torch.tensor([
            [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6]],
            [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6]],
            [[0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6], [0.6, 0.6, 0.6]],
        ])

        result = composed(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms

        composed = transforms.Compose([
            transforms.Resize((4, 4)),
        ])

        img = torch.tensor([
            [[0.2, 0.4], [0.6, 0.8]],
            [[0.1, 0.3], [0.5, 0.7]],
            [[0.0, 0.2], [0.4, 0.6]],
        ])

        result = composed(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms as transforms

        composed = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])

        img = Image.new('RGB', (4, 4), color=(128, 128, 128))
        result = composed(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms

        composed = transforms.Compose([
            transforms.CenterCrop(1),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ])

        img = torch.tensor([
            [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]],
            [[0.6, 0.6], [0.6, 0.6], [0.6, 0.6]],
            [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]],
        ])

        result = composed(img)
        """
    )
    obj.run(pytorch_code, ["result"])
