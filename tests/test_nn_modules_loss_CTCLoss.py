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

import pytest
from apibase import APIBase

obj = APIBase("torch.nn.modules.loss.CTCLoss")


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_1():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss()
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_2():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(blank=1)
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_3():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(reduction='none')
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_4():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(reduction='mean')
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_5():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(reduction='sum')
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_6():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(zero_infinity=True)
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_7():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_8():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(blank=0, zero_infinity=True, reduction='sum')
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_7
@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_9():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_7
@pytest.mark.skip(
    reason="CTCLoss results differ between PyTorch and Paddle framework implementations"
)
def test_case_10():
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
                                   [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]]], dtype=torch.float32)
        labels = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
        input_lengths = torch.tensor([4, 4], dtype=torch.int64)
        label_lengths = torch.tensor([3, 3], dtype=torch.int64)
        loss = torch.nn.modules.loss.CTCLoss(reduction='mean', zero_infinity=True, blank=0)
        result = loss(log_probs, labels, input_lengths, label_lengths)
        """
    )
    obj.run(pytorch_code, ["result"])
