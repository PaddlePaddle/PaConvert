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

obj = APIBase("torch.distributed.rpc.rpc_async")


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
        r = rpc.rpc_async(
            "worker1",
            torch.add,
            args=(torch.tensor(2), torch.tensor(3))
        )
        result = r.wait()
        rpc.shutdown()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support tensor in rpc_async",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import os
        import time
        import torch
        from torch.distributed import rpc
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        start = 25000
        end = 30000
        for port in range(start, end):
            try:
                s.bind(('localhost', port))
                s.close()
                break
            except socket.error:
                continue
        print("port: " + str(port))

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:' + str(port)
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.rpc_async(
            "worker1",
            time.time
        )
        result = r.wait()
        rpc.shutdown()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import os
        import time
        import torch
        import socket
        from torch.distributed import rpc
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        start = 25000
        end = 30000
        for port in range(start, end):
            try:
                s.bind(('localhost', port))
                s.close()
                break
            except socket.error:
                continue
        print("port: " + str(port))

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:' + str(port)
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.rpc_async(
            to = "worker1",
            func = time.time
        )
        result = r.wait()
        rpc.shutdown()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
