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
        from torch import optim
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
        # Forward pass.
        rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        # Optimizer.
        dist_optim = DistributedOptimizer(
            optim.SGD,
            [rref1, rref2],
            lr=0.05,
        )
        rpc.shutdown()

        """
    )
    obj.run(
        pytorch_code, 
        ["result"],
        unsupport=True,
        reason="paddle does not support tensor in DistributedOptimizer",)
