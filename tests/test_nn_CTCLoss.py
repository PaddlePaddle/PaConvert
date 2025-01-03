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

import textwrap

from apibase import APIBase

obj = APIBase("torch.nn.CTCLoss")


def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        log_probs = torch.tensor([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
                                   [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],
                                  [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
                                   [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],
                                  [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
                                   [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],
                                  [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],
                                  [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
                                   [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([5, 5], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        result = torch.nn.CTCLoss(blank=0, reduction='mean')(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch log_softmax + ctc_loss is equivalent to paddle ctc_loss",
    )


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        T = 50
        C = 20
        N = 16
        S = 30
        S_min = 10
        input = torch.arange(T * N * C).to(dtype=torch.float32).reshape((T, N, C)).log_softmax(0).detach().requires_grad_()
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.int32)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int64)
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.int64)
        ctc_loss = torch.nn.CTCLoss()
        result = ctc_loss(input, target, input_lengths, target_lengths)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch log_softmax + ctc_loss is equivalent to paddle ctc_loss",
    )
