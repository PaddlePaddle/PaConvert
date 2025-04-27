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
from torch.distributed import P2POp

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

if rank == 0:
    send_tensor = torch.ones(3, 3).cuda()
    p2p_op = P2POp(dist.isend, send_tensor, peer=1)
else:
    recv_tensor = torch.empty(3, 3).cuda()
    p2p_op = P2POp(dist.irecv, recv_tensor, peer=0)


reqs = dist.batch_isend_irecv([p2p_op])

for req in reqs:
    req.wait()

if rank != 0:
    print(recv_tensor)
