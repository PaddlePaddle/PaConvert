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

obj = APIBase("torchvision.transforms.functional.to_grayscale")
img_obj = ImageAPIBase("torchvision.transforms.functional.to_grayscale")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (3, 3), color=(255, 0, 0))
        result = F.to_grayscale(img, num_output_channels=1)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGB', (4, 4), color=(0, 255, 0))
        result = F.to_grayscale(img, 3)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        result = F.to_grayscale(Image.new('HSV', (5, 5), color=(120, 100, 100)), num_output_channels=1)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('RGBA', (2, 4), color=(0, 0, 255, 128))
        result = F.to_grayscale(num_output_channels=1, img=img)
        """
    )
    img_obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        import torchvision.transforms.functional as F
        img = Image.new('CMYK', (3, 3), color=(0, 128, 128, 0))
        result = F.to_grayscale(img=img, num_output_channels=3)
        """
    )
    img_obj.run(pytorch_code, ["result"])
