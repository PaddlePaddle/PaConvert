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

obj = APIBase("torchvision.transforms.functional.normalize")
img_obj = ImageAPIBase("torchvision.transforms.functional.normalize")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        mean = 0.5, 0.5, 0.5
        std = [0.5, 0.5, 0.5]
        img = torch.tensor([
            [[0.5, 0.5],
             [0.5, 0.5]],
            [[0.5, 0.5],
             [0.5, 0.5]],
            [[0.5, 0.5],
             [0.5, 0.5]]
        ])
        result = F.normalize(img, mean=mean, std=std)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [[0.2, 0.4],
             [0.6, 0.8]]
        ])
        result = F.normalize(img, mean=0.5, std=[0.5])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = torch.tensor([
            [
                [[0.5, 0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.5, 0.5]],
                [[0.4, 0.4, 0.4, 0.4],
                 [0.4, 0.4, 0.4, 0.4],
                 [0.4, 0.4, 0.4, 0.4],
                 [0.4, 0.4, 0.4, 0.4]],
                [[0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3],
                 [0.3, 0.3, 0.3, 0.3]]
            ],
            [
                [[0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.6],
                 [0.6, 0.6, 0.6, 0.6]],
                [[0.7, 0.7, 0.7, 0.7],
                 [0.7, 0.7, 0.7, 0.7],
                 [0.7, 0.7, 0.7, 0.7],
                 [0.7, 0.7, 0.7, 0.7]],
                [[0.8, 0.8, 0.8, 0.8],
                 [0.8, 0.8, 0.8, 0.8],
                 [0.8, 0.8, 0.8, 0.8],
                 [0.8, 0.8, 0.8, 0.8]]
            ]
        ])
        result = F.normalize(tensor=img, std=std, mean=mean)
        """
    )
    obj.run(pytorch_code, ["result"])
