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

obj = APIBase("torch.distributed.all_gather_object")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.distributed as dist
        result = None
        if torch.cuda.is_available():
            dist.init_process_group("nccl", init_method='tcp://127.0.0.1:23456', rank=0, world_size=3)
            gather_objects = ["foo", 12, {1: 2}]
            output = [None for _ in gather_objects]
            dist.all_gather_object(output, gather_objects[dist.get_rank()])
            result = True
        """
    )
    obj.run(pytorch_code, ["result"])
