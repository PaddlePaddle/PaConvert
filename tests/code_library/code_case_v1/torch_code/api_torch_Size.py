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
print(torch.Size([2, 8, 64, 64]))
print("#########################case2#########################")
assert torch.randn(6, 5, 7).size() == torch.Size([6, 5, 7])
print("#########################case3#########################")
out = torch.Size([6, 5, 7])
shape_nchw = torch.Size([6, 5, 7])
assert out == torch.Size(shape_nchw)
print("#########################case4#########################")
print(torch.Size([1]))
print("#########################case5#########################")
shape = torch.Size([1])
