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

obj = APIBase("torchvision.transforms.RandomErasing")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomErasing, InterpolationMode

        torch.manual_seed(0)

        transform = RandomErasing(p=0.3, scale=(0.2, 0.4), ratio=(0.3, 3.3), value=0, inplace=False)

        img = torch.tensor([
            [
                [0, 20, 30, 40],
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

        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomErasing, InterpolationMode

        torch.manual_seed(1)

        transform = RandomErasing(0.0, (0.1, 0.2), (0.5, 2.0), 1, True)

        img = torch.tensor([
            [
                [5, 10, 15, 20, 25],
                [30, 35, 40, 45, 50],
                [55, 60, 65, 70, 75],
                [80, 85, 90, 95, 100],
                [105, 110, 115, 120, 125]
            ]
        ], dtype=torch.float)

        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomErasing, InterpolationMode

        torch.manual_seed(2)

        scale = (0.1, 0.3)
        ratio = (0.5, 2.0)
        value = (255, 255, 255)
        inplace = False
        transform = RandomErasing(p=0.5, scale=scale, ratio=ratio, value=value, inplace=inplace)

        img = torch.tensor([
            [
                [100, 150, 200],
                [150, 200, 250],
                [200, 250, 300]
            ],
            [
                [110, 160, 210],
                [160, 210, 260],
                [210, 260, 310]
            ],
            [
                [120, 170, 220],
                [170, 220, 270],
                [220, 270, 320]
            ]
        ], dtype=torch.float)

        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.transforms import RandomErasing, InterpolationMode

        torch.manual_seed(4)

        transform = RandomErasing(0.3, (0.05, 0.2), ratio=(0.3, 3.3), value=[0, 0, 0], inplace=True)

        img = torch.tensor([
            [
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [110, 120, 130, 140, 150],
                    [160, 170, 180, 190, 200],
                    [210, 220, 230, 240, 250]
                ],
                [
                    [255, 0, 255, 0, 255],
                    [0, 255, 0, 255, 0],
                    [255, 0, 255, 0, 255],
                    [0, 255, 0, 255, 0],
                    [255, 0, 255, 0, 255]
                ],
                [
                    [128, 128, 128, 128, 128],
                    [64, 64, 64, 64, 64],
                    [32, 32, 32, 32, 32],
                    [16, 16, 16, 16, 16],
                    [8, 8, 8, 8, 8]
                ]
            ],
            [
                [
                    [5, 10, 15, 20, 25],
                    [30, 35, 40, 45, 50],
                    [55, 60, 65, 70, 75],
                    [80, 85, 90, 95, 100],
                    [105, 110, 115, 120, 125]
                ],
                [
                    [0, 255, 0, 255, 0],
                    [255, 0, 255, 0, 255],
                    [0, 255, 0, 255, 0],
                    [255, 0, 255, 0, 255],
                    [0, 255, 0, 255, 0]
                ],
                [
                    [8, 16, 24, 32, 40],
                    [16, 32, 48, 64, 80],
                    [24, 48, 72, 96, 120],
                    [32, 64, 96, 128, 160],
                    [40, 80, 120, 160, 200]
                ]
            ]
        ], dtype=torch.float)

        result = transform(img)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
