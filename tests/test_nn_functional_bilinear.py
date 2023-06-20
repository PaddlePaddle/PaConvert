# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.functional.bilinear")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input1 = torch.tensor([[ 0.78519595, -0.22123627, -0.77207279],
                            [-1.72364950, -1.17815387,  0.95774752],
                            [-2.06469393,  1.74343407, -0.62424982]])
        input2 = torch.tensor([[-1.58770573,  0.36451983, -0.01229221,  0.88867152],
                            [ 0.63016212,  0.75923228, -1.42190886, -0.91600877],
                            [ 0.56267124, -0.35368094,  1.67497897, -0.76758206]])
        weight = torch.tensor([[[-0.26903051, -1.42116427, -0.67254508,  1.11618853],
                                [-0.96080893, -0.50334930, -0.95949239, -1.28214407],
                                [ 0.18504630,  0.09448750, -0.20859139,  1.60624814]],

                                [[ 0.32964826, -0.47029141,  0.04190124, -0.02686183],
                                [-0.05973548,  1.47110248, -0.59110558, -1.87694490],
                                [ 0.65067267,  0.28164786, -1.04381704,  2.01811719]]])
        bias = torch.tensor([-0.33773780,  1.08489835])
        result = F.bilinear(input1, input2, weight, bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input1 = torch.tensor([[ 0.78519595, -0.22123627, -0.77207279],
                            [-1.72364950, -1.17815387,  0.95774752],
                            [-2.06469393,  1.74343407, -0.62424982]])
        input2 = torch.tensor([[-1.58770573,  0.36451983, -0.01229221,  0.88867152],
                            [ 0.63016212,  0.75923228, -1.42190886, -0.91600877],
                            [ 0.56267124, -0.35368094,  1.67497897, -0.76758206]])
        weight = torch.tensor([[[-0.26903051, -1.42116427, -0.67254508,  1.11618853],
                                [-0.96080893, -0.50334930, -0.95949239, -1.28214407],
                                [ 0.18504630,  0.09448750, -0.20859139,  1.60624814]],

                                [[ 0.32964826, -0.47029141,  0.04190124, -0.02686183],
                                [-0.05973548,  1.47110248, -0.59110558, -1.87694490],
                                [ 0.65067267,  0.28164786, -1.04381704,  2.01811719]]])
        result = F.bilinear(input1, input2, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input1 = torch.tensor([[ 0.78519595, -0.22123627, -0.77207279],
                            [-1.72364950, -1.17815387,  0.95774752],
                            [-2.06469393,  1.74343407, -0.62424982]])
        input2 = torch.tensor([[-1.58770573,  0.36451983, -0.01229221,  0.88867152],
                            [ 0.63016212,  0.75923228, -1.42190886, -0.91600877],
                            [ 0.56267124, -0.35368094,  1.67497897, -0.76758206]])
        weight = torch.tensor([[[-0.26903051, -1.42116427, -0.67254508,  1.11618853],
                                [-0.96080893, -0.50334930, -0.95949239, -1.28214407],
                                [ 0.18504630,  0.09448750, -0.20859139,  1.60624814]],

                                [[ 0.32964826, -0.47029141,  0.04190124, -0.02686183],
                                [-0.05973548,  1.47110248, -0.59110558, -1.87694490],
                                [ 0.65067267,  0.28164786, -1.04381704,  2.01811719]]])
        bias = torch.tensor([-0.33773780,  1.08489835])
        result = F.bilinear(input1=input1, input2=input2, weight=weight, bias=bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input1 = torch.tensor([[ 0.78519595, -0.22123627, -0.77207279],
                            [-1.72364950, -1.17815387,  0.95774752],
                            [-2.06469393,  1.74343407, -0.62424982]])
        input2 = torch.tensor([[-1.58770573,  0.36451983, -0.01229221,  0.88867152],
                            [ 0.63016212,  0.75923228, -1.42190886, -0.91600877],
                            [ 0.56267124, -0.35368094,  1.67497897, -0.76758206]])
        weight = torch.tensor([[[-0.26903051, -1.42116427, -0.67254508,  1.11618853],
                                [-0.96080893, -0.50334930, -0.95949239, -1.28214407],
                                [ 0.18504630,  0.09448750, -0.20859139,  1.60624814]],

                                [[ 0.32964826, -0.47029141,  0.04190124, -0.02686183],
                                [-0.05973548,  1.47110248, -0.59110558, -1.87694490],
                                [ 0.65067267,  0.28164786, -1.04381704,  2.01811719]]])

        result = F.bilinear(input1=input1, input2=input2, weight=weight, bias = torch.tensor([-0.33773780,  1.08489835]))
        """
    )
    obj.run(pytorch_code, ["result"])
