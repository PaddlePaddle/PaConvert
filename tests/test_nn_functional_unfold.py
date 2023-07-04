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

obj = APIBase("torch.nn.functional.unfold")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )
        result = torch.nn.functional.unfold(input,kernel_size=(2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )

        result = torch.nn.functional.unfold(input,kernel_size=(2, 3),padding=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )

        result = torch.nn.functional.unfold(input,kernel_size=(2, 3),padding=(2,2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )

        result = torch.nn.functional.unfold(input,kernel_size=(2, 3),dilation=(1,1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )

        result = torch.nn.functional.unfold(input,kernel_size=(2, 3),stride=(2,2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )

        result = torch.nn.functional.unfold(input,kernel_size=(2, 3),stride=(2,2),dilation=(1,1),padding=(2,2))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )
        kernel_size=(2,2)
        stride=(2,2)
        dilation=(1,1)
        padding=(1,1)

        result = torch.nn.functional.unfold(input,kernel_size=kernel_size,stride=stride,dilation=dilation,padding=padding)

        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor(
             [[[[0.5018016,  0.71745074, 0.02612579, 0.04813039],
                [0.14209914, 0.45702428, 0.06756079, 0.73914427],
                [0.35131782, 0.03954667, 0.1214295, 0.25422984]],

               [[0.3040169,  0.650879,   0.29451096, 0.4443251 ],
               [0.00550938, 0.38386834, 0.48462474, 0.49691153],
               [0.9952472,  0.05594945, 0.6351355,  0.6343607 ]]],


              [[[0.37795508, 0.63193935, 0.19294626, 0.77718097],
               [0.785048,   0.67698157, 0.6636463,  0.63043   ],
               [0.3141495,  0.48402798, 0.43465394, 0.52195907]],

              [[0.8227394,  0.47486508, 0.41936857, 0.08142513],
              [0.518088,   0.5427299,  0.9754643,  0.58517313],
              [0.0467307,  0.18104774, 0.9747845,  0.84306306]]]]
              )


        result = torch.nn.functional.unfold(input,kernel_size=[2,2],stride=[2,2],dilation=[1,1],padding=[1,1])


        """
    )
    obj.run(pytorch_code, ["result"])
