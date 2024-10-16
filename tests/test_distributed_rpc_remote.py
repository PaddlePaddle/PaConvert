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

obj = APIBase("torch.distributed.rpc.remote", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import os
        import torch
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

        def add(a, b):
            return a+b

        from torch.distributed import rpc
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:' + str(port)
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.remote(
            "worker1",
            add,
            args=(2, 3)
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

        def add(a, b):
            return a+b

        from torch.distributed import rpc
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:' + str(port)
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.remote(
            to="worker1",
            func=torch.add,
            args=(2, 3),
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


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import os
        import torch
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

        def add(a, b):
            return a+b

        from torch.distributed import rpc
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:' + str(port)
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.remote(
            to="worker1",
            func=add,
            args=(2, 3),
            timeout=-1,
            kwargs=None
        )
        result = r.to_here()
        rpc.shutdown()
        """
    )
    obj.run(
        pytorch_code,
        ["result"]
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import os
        import torch
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

        def add(a, b):
            return a+b

        from torch.distributed import rpc
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['PADDLE_MASTER_ENDPOINT'] = 'localhost:' + str(port)
        rpc.init_rpc(
            "worker1",
            rank=0,
            world_size=1
        )
        r = rpc.remote(
            to="worker1",
            func=add,
            args=None,
            timeout=-1,
            kwargs={"a": 2, "b": 3}
        )
        result = r.to_here()
        rpc.shutdown()
        """
    )
    obj.run(
        pytorch_code,
        ["result"]
    )
