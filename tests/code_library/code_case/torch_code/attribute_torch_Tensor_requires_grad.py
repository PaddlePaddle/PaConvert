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
data = torch.tensor([23.0, 32.0, 43.0])
if not data.requires_grad:
    print(1)
print("#########################case2#########################")
print(data.requires_grad)
print("#########################case3#########################")
data.requires_grad = False
print("#########################case4#########################")
requires_grad = data.requires_grad
print("#########################case5#########################")
data = torch.tensor([23.0, 32.0, 43.0], requires_grad=data.requires_grad)
print("#########################case6#########################")
print(data.requires_grad == False)
print("#########################case7#########################")
print(not data.requires_grad)
print("#########################case8#########################")
print("{} , {}".format("1", str(data.requires_grad)))
print("#########################case9#########################")


def test():
    return True


data.requires_grad = test()
print("#########################case10#########################")
z = (True, False, True)
a, data.requires_grad, c = z
print(data.requires_grad)
