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

obj = APIBase("torch.distributed.rpc.remote")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import os
        import torch
        from torch.distributed import rpc
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:29501'
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.remote(
            "worker1",
            torch.add,
            args=(torch.tensor(2), torch.tensor(3))
        )
        result = r.to_here()
        rpc.shutdown()
        """
    )
    obj.run(
        pytorch_code,
        ["result"]
    )

def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import os
        import torch
        from torch.distributed import rpc
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:29501'
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.remote(
            to="worker1",
            func=torch.add,
            args=(torch.tensor(2), torch.tensor(3)),
            kwargs=None,
            timeout=-1
        )
        result = r.to_here()
        rpc.shutdown()
        """
    )
    obj.run(
        pytorch_code,
        ["result"]
    )
