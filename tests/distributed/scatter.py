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
import common
import torch
import torch.distributed as dist

common.init_env()

tensor_size = 2
t_ones = torch.ones(tensor_size).cuda()
t_fives = torch.ones(tensor_size).cuda() * 5
output_tensor = torch.zeros(tensor_size).cuda()
if dist.get_rank() == 0:
    scatter_list = [t_ones, t_fives]
else:
    scatter_list = None
dist.scatter(output_tensor, scatter_list, src=0, group=None, async_op=False)

print("out:")
print(output_tensor)
common.dump_output(output_tensor)
