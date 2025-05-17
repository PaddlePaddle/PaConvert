# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#
import os

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)


input = torch.tensor(
    [[rank * 4, rank * 4 + 1], [rank * 4 + 2, rank * 4 + 3]], device=f"cuda:{rank}"
)

output = torch.empty_like(input)


dist.all_to_all_single(output, input)

if rank == 0:
    print(output)
    torch.save(output, os.environ["DUMP_FILE"])
