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

data = torch.arange(2, dtype=torch.int64).cuda() + 1 + 2 * rank

world_size = dist.get_world_size()
output_tensor = torch.empty(
    (world_size * data.size(0),), dtype=data.dtype, device=data.device
)


dist.all_gather_into_tensor(output_tensor, data)

print(f"Rank {rank} output tensor: {output_tensor}")

if rank == 0:
    torch.save(output_tensor, os.environ["DUMP_FILE"])
