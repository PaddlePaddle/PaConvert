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
#
import os

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

out_tensor_list = [
    torch.empty([2, 3], dtype=torch.int64).cuda(),
    torch.empty([2, 3], dtype=torch.int64).cuda(),
]
if dist.get_rank() == 0:
    data1 = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
    data2 = torch.tensor([[7, 8, 9], [10, 11, 12]]).cuda()
else:
    data1 = torch.tensor([[13, 14, 15], [16, 17, 18]]).cuda()
    data2 = torch.tensor([[19, 20, 21], [22, 23, 24]]).cuda()
dist.all_to_all(
    output_tensor_list=out_tensor_list,
    input_tensor_list=[data1, data2],
    group=None,
    async_op=False,
)

if dist.get_rank() == 0:
    print(out_tensor_list)
    torch.save(out_tensor_list, os.environ["DUMP_FILE"])
