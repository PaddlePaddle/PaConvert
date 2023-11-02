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
import torch.distributed as dist

common.init_env()

if dist.get_rank() == 0:
    object_list = [{"foo": [1, 2, 3]}]
else:
    object_list = [{"bar": [4, 5, 6]}]
dist.broadcast_object_list(object_list, src=1)

assert object_list == [{"bar": [4, 5, 6]}]
print(object_list)
common.dump_output(object_list)
