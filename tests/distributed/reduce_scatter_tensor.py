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

input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).cuda()
output = torch.empty(1, 2, dtype=torch.float32).cuda()
dist.reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM)

if rank == 0:
    print(output)
    torch.save(output, os.environ["DUMP_FILE"])
