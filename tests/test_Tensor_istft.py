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

obj = APIBase("torch.Tensor.istft")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[ (5.975718021392822+0j)                  ,
           (5.975718021392822+0j)                  ,
           (5.341437339782715+0j)                  ,
           (5.404394626617432+0j)                  ,
           (5.404394626617432+0j)                  ],
         [ (0.0629572868347168+0j)                 ,
           0.0629572868347168j                     ,
          (-0.0629572868347168-0.6342806816101074j),
           (0.6342806816101074+0j)                 ,
           0.6342806816101074j                     ],
         [(-0.4979677200317383+0j)                 ,
           (0.4979677200317383+0j)                 ,
           (0.13631296157836914+0j)                ,
          (-0.19927024841308594+0j)                ,
           (0.19927024841308594+0j)                ]]])
        result = x.istft(n_fft=4)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[ (5.975718021392822+0j)                  ,
           (5.975718021392822+0j)                  ,
           (5.341437339782715+0j)                  ,
           (5.404394626617432+0j)                  ,
           (5.404394626617432+0j)                  ],
         [ (0.0629572868347168+0j)                 ,
           0.0629572868347168j                     ,
          (-0.0629572868347168-0.6342806816101074j),
           (0.6342806816101074+0j)                 ,
           0.6342806816101074j                     ],
         [(-0.4979677200317383+0j)                 ,
           (0.4979677200317383+0j)                 ,
           (0.13631296157836914+0j)                ,
          (-0.19927024841308594+0j)                ,
           (0.19927024841308594+0j)                ]]])
        result = x.istft(n_fft=4, center=False)
        """
    )
    obj.run(pytorch_code, ["result"])
