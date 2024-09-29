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

obj = APIBase("torchvision.transforms.CenterCrop")
img_obj = ImageAPIBase("torchvision.transforms.CenterCrop")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms
        img = torch.tensor([[[0.5, 0.5], [0.5, 0.5]],
                             [[0.5, 0.5], [0.5, 0.5]],
                             [[0.5, 0.5], [0.5, 0.5]]])
        center_crop = transforms.CenterCrop((1, 1))
        result = center_crop(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms
        img = torch.tensor([[[0.1, 0.4], [0.7, 1.0]],
                             [[0.2, 0.5], [0.8, 1.0]],
                             [[0.3, 0.6], [0.9, 1.0]]])
        center_crop = transforms.CenterCrop([1, 1])
        result = center_crop(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms as transforms
        img = Image.new('RGB', (4, 4), color=(100, 100, 100))
        center_crop = transforms.CenterCrop(2)
        result = center_crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms
        img = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7, 0.8]],
                             [[0.9, 1.0, 1.1, 1.2],
                              [1.3, 1.4, 1.5, 1.6]]])
        center_crop = transforms.CenterCrop((2, 2))
        result = center_crop(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms as transforms
        img = Image.new('RGB', (3, 3), color=(50, 100, 150))
        center_crop = transforms.CenterCrop((2, 2))
        result = center_crop(img)
        """
    )
    img_obj.run(pytorch_code, ["result"])
