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
        from torch.optim import SGD
        from torch.distributed.optim import DistributedOptimizer

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:29501'
        # 初始化RPC
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.param = nn.Linear(10, 1)

            def forward(self, x):
                return self.param(x)
        
        # 初始化
        data = torch.randn(1, 10)
        target = torch.randn(1, 1)
        model = SimpleModel()

        # 创建远程模型
        remote_model_rref = rpc.remote("worker1", model, args=(data))
        # 创建分布式优化器
        optimizer_class = SGD
        optimizer_args = (params_rref,)
        optimizer_kwargs = {'lr': 0.01}
        optimizer = DistributedOptimizer(optimizer_class, params_rref, *optimizer_args, **optimizer_kwargs)

        # 输出
        output = remote_model_rref.to_here()
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(worker1)
        rpc.shutdown()

        result = 1
        """
    )
    obj.run(pytorch_code, ["result"])
