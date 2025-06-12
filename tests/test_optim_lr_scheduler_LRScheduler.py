# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import textwrap

from apibase import APIBase

obj = APIBase("torch.optim.lr_scheduler.LRScheduler")


# TODO: fix torch.atleast bug, which not support input list/tuple
def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import math

        class CosineAnnealingScheduler(optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
                self.T_max = T_max
                self.eta_min = eta_min
                super().__init__(optimizer, last_epoch)

            def get_lr(self):
                return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                        for base_lr in self.base_lrs]


        model = nn.Linear(10, 1)

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineAnnealingScheduler(optimizer, T_max=10)

        inputs = torch.randn(100, 10)
        targets = torch.randn(100, 1)

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

        result1 = model.weight
        result2 = model.bias
        """
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)
