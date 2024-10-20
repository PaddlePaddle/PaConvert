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

obj = APIBase("torchvision.transforms.ToTensor")
img_obj = ImageAPIBase("torchvision.transforms.ToTensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        from torchvision.transforms import ToTensor
        import torch

        transform = ToTensor()
        img = Image.new('RGB', (3, 3), color=(255, 0, 0))
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3, 3)
        """
    )
    img_obj.run(pytorch_code, [])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        from torchvision.transforms import ToTensor
        import torch

        transform = ToTensor()
        img = Image.new('L', (4, 4), color=128)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 4, 4)
        """
    )
    img_obj.run(pytorch_code, [])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        from PIL import Image
        from torchvision.transforms import ToTensor
        import torch

        transform = ToTensor()
        img_np = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]],
            [[255, 0, 255], [192, 192, 192], [128, 128, 128], [64, 64, 64], [0, 0, 0]],
            [[255, 165, 0], [0, 128, 128], [128, 0, 128], [128, 128, 0], [0, 0, 128]],
            [[75, 0, 130], [238, 130, 238], [245, 222, 179], [255, 105, 180], [0, 255, 127]],
            [[255, 20, 147], [173, 216, 230], [144, 238, 144], [255, 182, 193], [64, 224, 208]]
        ], dtype=np.uint8)
        img = Image.fromarray(img_np)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 5, 5)
        """
    )
    obj.run(pytorch_code, [])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from PIL import Image
        from torchvision.transforms import ToTensor
        import torch

        transform = ToTensor()
        img = Image.new('RGBA', (2, 4), color=(0, 0, 255, 128))
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 2, 4)
        """
    )
    img_obj.run(pytorch_code, [])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        from PIL import Image
        from torchvision.transforms import ToTensor
        import torch

        transform = ToTensor()
        img_np = np.array([
            [0, 128, 255],
            [64, 192, 32],
            [16, 240, 80]
        ], dtype=np.uint8)
        img = Image.fromarray(img_np)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3, 1)
        """
    )
    obj.run(pytorch_code, [])