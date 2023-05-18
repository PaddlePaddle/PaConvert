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
import torch.nn as nn

print("#########################case1#########################")
m = nn.InstanceNorm3d(100)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("#########################case2#########################")
m = nn.InstanceNorm3d(100, affine=True)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("#########################case3#########################")
m = nn.InstanceNorm3d(100, affine=False)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("#########################case4#########################")
m = nn.InstanceNorm3d(100, affine=True, momentum=0.1)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("#########################case5#########################")
m = nn.InstanceNorm3d(100, affine=False, momentum=0.1)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
