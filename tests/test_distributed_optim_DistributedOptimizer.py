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
from optimizer_helper import generate_optimizer_test_code

obj = APIBase("torch.distributed.optim.DistributedOptimizer")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import os
        import torch
        import torch.distributed as dist
        import torch.nn as nn
        import torch.distributed.rpc as rpc
        from torch.distributed.optim import DistributedOptimizer

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:29501'
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )

        data = nn.Parameter(torch.tensor([[-0.4229,  1.2159, -1.3944,  0.8764, -2.5841, -2.1045, -0.7999,  0.1856,
                0.6989,  0.3954]]), requires_grad=True)
        rref1 = rpc.remote("worker1", torch.add, args=(data, 3))
        target = torch.tensor([[1.1352]])
        Loss_fuc = torch.nn.MSELoss()

        optimizer = DistributedOptimizer(
                torch.optim.SGD,
                [rref1],
                lr=0.01
            )

        # 打印损失
        print(f"Iteration {i}, Loss: {loss.item()}")
        rpc.shutdown()
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])
