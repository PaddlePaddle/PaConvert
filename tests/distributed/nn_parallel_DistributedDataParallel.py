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

import os

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

model = torch.nn.Linear(1, 1, bias=False).to("cuda")
model = torch.nn.parallel.DistributedDataParallel(model)

print(model)
