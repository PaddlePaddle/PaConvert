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

obj = APIBase("torchvision.transforms.Normalize")
img_obj = ImageAPIBase("torchvision.transforms.Normalize")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img = torch.tensor([
            [[0.5, 0.5],
             [0.5, 0.5]],

            [[0.5, 0.5],
             [0.5, 0.5]],

            [[0.5, 0.5],
             [0.5, 0.5]]
        ])

        result = normalize(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

        img = torch.tensor([
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]],

            [[10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0],
             [16.0, 17.0, 18.0]],

            [[19.0, 20.0, 21.0],
             [22.0, 23.0, 24.0],
             [25.0, 26.0, 27.0]]
        ])

        result = normalize(img)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

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

        result = normalize(img)
        """
    )
    obj.run(pytorch_code, ["result"])
