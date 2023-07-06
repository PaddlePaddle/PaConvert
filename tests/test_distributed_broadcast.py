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

obj = APIBase("torch.distributed.broadcast")


def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([[1, 2, 3, 4],[5,6,7,8]])
        torch.distributed.broadcast(data, src=0)
        result=True
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle need init_parallel_env() paconvert currently not supported for transcription ",
    )


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([[1, 2, 3, 4],[5,6,7,8]])
        torch.distributed.broadcast(data, src=0,group=1)
        result=True
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle need init_parallel_env() paconvert currently not supported for transcription",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([[1, 2, 3, 4],[5,6,7,8]])
        torch.distributed.broadcast(data, src=0,async_op=True)
        result=True
        """
    )
    obj.run(
        pytorch_code, ["result"], unsupport=True, reason="paddle unsupport async_op"
    )
