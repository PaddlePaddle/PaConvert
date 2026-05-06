# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.distributed.init_process_group")


def test_case_1():
    """Bare call — kwargs forwarded as-is via ChangePrefixMatcher."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.distributed.init_process_group(backend='nccl')
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        paddle.distributed.init_process_group(backend="nccl")
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)


def test_case_2():
    """All torch kwargs preserved (paddle accepts and ignores the unused ones)."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:23456',
            world_size=4,
            rank=0,
        )
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        paddle.distributed.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:23456", world_size=4, rank=0
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)


def test_case_3():
    """Module-style import."""
    pytorch_code = textwrap.dedent(
        """
        import torch.distributed as dist
        dist.init_process_group(backend='gloo')
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        paddle.distributed.init_process_group(backend="gloo")
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)
