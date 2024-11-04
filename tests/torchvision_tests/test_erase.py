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

obj = APIBase("torchvision.transforms.functional.erase")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        i, j, h, w = 1, 1, 2, 2
        v = 0.0
        inplace = False
        img = torch.tensor([
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160]
            ],
            [
                [15, 25, 35, 45],
                [55, 65, 75, 85],
                [95, 105, 115, 125],
                [135, 145, 155, 165]
            ],
            [
                [20, 30, 40, 50],
                [60, 70, 80, 90],
                [100, 110, 120, 130],
                [140, 150, 160, 170]
            ]
        ], dtype=torch.float)
        result = F.erase(img, i, j, h, w, v, inplace)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        i, j, h, w = 0, 0, 3, 3
        v = 1.0
        inplace = True
        img = torch.tensor([
            [
                [5, 10, 15, 20, 25],
                [30, 35, 40, 45, 50],
                [55, 60, 65, 70, 75],
                [80, 85, 90, 95, 100],
                [105, 110, 115, 120, 125]
            ]
        ], dtype=torch.float)
        result = F.erase(img=img, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        img = torch.tensor([
            [
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160]
            ],
            [
                [15, 25, 35, 45],
                [55, 65, 75, 85],
                [95, 105, 115, 125],
                [135, 145, 155, 165]
            ],
            [
                [20, 30, 40, 50],
                [60, 70, 80, 90],
                [100, 110, 120, 130],
                [140, 150, 160, 170]
            ]
        ], dtype=torch.float)
        result = F.erase(img, 3, 3, 2, 2, 0.5)
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
                [1, 2, 3, 4],
                [5, 6, 7, 8]
            ],
            [
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ],
            [
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ],
            [
                [25, 26, 27, 28],
                [29, 30, 31, 32]
            ]
        ], dtype=torch.float)
        v = torch.tensor([10.0, 20.0, 30.0, 40.0])
        result = F.erase(img, 0, 0, 1, 4, torch.tensor([10.0, 20.0, 30.0, 40.0]), False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.transforms.functional as F
        i, j, h, w = 0, 0, 3, 3
        v = 1.0
        inplace = True
        img = torch.tensor([
            [
                [5, 10, 15, 20, 25],
                [30, 35, 40, 45, 50],
                [55, 60, 65, 70, 75],
                [80, 85, 90, 95, 100],
                [105, 110, 115, 120, 125]
            ]
        ], dtype=torch.float)
        result = F.erase(img=img, h=h, w=w, v=v, i=i, j=j, inplace=inplace)
        """
    )
    obj.run(pytorch_code, ["result"])
