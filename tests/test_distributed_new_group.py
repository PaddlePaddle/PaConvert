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

obj = APIBase("torch.distributed.new_group")


def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.distributed.init_process_group(
            "nccl",
            init_method="tcp://127.0.0.1:23456",
            rank=0,
            world_size=1
        )
        torch.distributed.new_group(list(range(1)))
        result=True
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.distributed.init_process_group(
            "nccl",
            init_method="tcp://127.0.0.1:23456",
            rank=0,
            world_size=1
        )
        torch.distributed.new_group(list(range(1)),pg_options=None)
        result=True
        """
    )
    obj.run(pytorch_code, ["result"])
