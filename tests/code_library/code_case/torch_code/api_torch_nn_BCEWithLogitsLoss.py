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
loss = torch.nn.BCEWithLogitsLoss(reduction="none")
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("#########################case2#########################")
loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0, 0.2, 0.2]), reduction="none")
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("#########################case3#########################")
loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones([3]))
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("#########################case4#########################")
loss = nn.BCEWithLogitsLoss(size_average=True)
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("#########################case5#########################")
loss = nn.BCEWithLogitsLoss()
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
