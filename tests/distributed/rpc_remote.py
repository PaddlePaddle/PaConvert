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
#
import os

import torch
from torch.distributed import rpc

if "PADDLE_GLOBAL_RANK" in os.environ and "RANK" not in os.environ:
    os.environ["RANK"] = os.environ["PADDLE_GLOBAL_RANK"]
if "PADDLE_GLOBAL_SIZE" in os.environ and "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = os.environ["PADDLE_GLOBAL_SIZE"]
if "PADDLE_MASTER" in os.environ and "PADDLE_MASTER_ENDPOINT" not in os.environ:
    os.environ["PADDLE_MASTER_ENDPOINT"] = os.environ["PADDLE_MASTER"]

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
worker_name = f"worker{rank}"

rpc.init_rpc(worker_name, rank=rank, world_size=world_size)
remote = rpc.remote("worker0", min, args=(2, 1))
result = remote.to_here()
rpc.shutdown()

if rank == 0:
    print(result)
    torch.save(result, os.environ["DUMP_FILE"])
