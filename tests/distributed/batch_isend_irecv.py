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

import torch
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.zeros(10).cuda()

if rank == 0:
    op_list = [P2POp(dist.isend, tensor, 1)]
elif rank == 1:
    op_list = [P2POp(dist.irecv, tensor, 0)]

work_list = batch_isend_irecv(op_list)

for work in work_list:
    work.wait()
