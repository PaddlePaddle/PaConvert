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

obj = APIBase("torchvision.transforms.functional.adjust_hue")
img_obj = ImageAPIBase("torchvision.transforms.functional.adjust_hue")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([[[0.5, 0.5], [0.5, 0.5]],
                             [[0.5, 0.5], [0.5, 0.5]],
                             [[0.5, 0.5], [0.5, 0.5]]])
        hue_factor = 0.5
        result = F.adjust_hue(img, hue_factor)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([[[0.1, 0.4], [0.7, 1.0]],
                             [[0.2, 0.5], [0.8, 1.0]],
                             [[0.3, 0.6], [0.9, 1.0]]])
        result = F.adjust_hue(img, 0.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (2, 2), color=(100, 100, 100))
        result = F.adjust_hue(img, 0.25)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        result = F.adjust_hue(torch.tensor([[[0.0, 0.2], [0.3, 0.4]],
                                              [[0.5, 0.6], [0.7, 0.8]],
                                              [[0.9, 1.0], [1.0, 1.0]]]), -0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (2, 2), color=(50, 100, 150))
        result = F.adjust_hue(img, -0.25)
        """
    )
    img_obj.run(pytorch_code, ["result"])
