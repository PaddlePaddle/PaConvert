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

import torch

print("#########################case1#########################")
cpu = torch.device("cpu")
a = torch.randn(2, 3)
c = torch.randn(2, 3, dtype=torch.float64, device=cpu)
b = a.to(cpu, non_blocking=False, copy=False)
print("#########################case2#########################")
b = a.to("cpu")
print("#########################case3#########################")
b = a.to(device=cpu, dtype=torch.float64)
print("#########################case4#########################")
b = a.to(torch.float64)
print("#########################case5#########################")
b = a.to(dtype=torch.float64)
print("#########################case6#########################")
b = a.to(c)
print("#########################case7#########################")
a = a.to(torch.half)
print("#########################case8#########################")
table = a
b = a.to(table.device)
print("#########################case9#########################")
b = a.to(torch.float32)

print("#########################case10#########################")
device = torch.device("cpu")
b = torch.tensor([-1]).to(torch.bool)

print("#########################case11#########################")
dtype = torch.float32
b = a.to(dtype=dtype)

print("#########################case12#########################")
b = a.to(torch.device("cpu"))
