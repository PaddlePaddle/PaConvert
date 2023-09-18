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

obj = APIBase("torch.linalg.matrix_rank")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.eye(10)
        A[0, 0] = 0
        result = torch.linalg.matrix_rank(A)
        """
    )
    # NOTE: torch dtype is int64, paddle dtype is int32
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[[[-1.1079, -0.4803,  0.2296],
          [ 0.3198, -0.2976, -0.0585],
          [-1.6931, -0.3353, -0.2893]],
         [[ 1.9757, -0.5959, -1.2041],
          [ 0.8443, -0.4916, -1.6574],
          [-0.2654, -1.0447, -0.8138]],
         [[-0.4111,  1.0973,  0.2275],
          [ 1.1851,  1.8233,  0.8187],
          [-1.4107, -0.5473,  1.1431]],
         [[ 0.0327, -0.8295,  0.0457],
          [-0.6286, -0.2507,  0.7292],
          [ 0.4075, -1.3918, -0.5015]]],
        [[[-2.1256,  0.9310,  1.0743],
          [ 1.9577, -0.1513,  0.1668],
          [-0.1404,  1.6647,  0.7108]],
         [[ 0.9001,  1.6930, -0.4966],
          [-1.0432, -1.0742,  1.2273],
          [-0.2711, -0.4740, -0.6381]],
         [[-1.3099, -1.7540,  0.5443],
          [ 0.3565, -2.3821,  0.8638],
          [-1.3840,  0.8216,  0.2761]],
         [[-0.5989, -0.4732,  1.3252],
          [-0.7614,  1.0493,  0.8488],
          [-0.1300,  0.1287,  0.6234]]]])
        result = torch.linalg.matrix_rank(A, atol=torch.tensor(1.0), rtol=torch.tensor(0.0),  hermitian=True)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[[[-1.1079, -0.4803,  0.2296],
          [ 0.3198, -0.2976, -0.0585],
          [-1.6931, -0.3353, -0.2893]],
         [[ 1.9757, -0.5959, -1.2041],
          [ 0.8443, -0.4916, -1.6574],
          [-0.2654, -1.0447, -0.8138]],
         [[-0.4111,  1.0973,  0.2275],
          [ 1.1851,  1.8233,  0.8187],
          [-1.4107, -0.5473,  1.1431]],
         [[ 0.0327, -0.8295,  0.0457],
          [-0.6286, -0.2507,  0.7292],
          [ 0.4075, -1.3918, -0.5015]]],
        [[[-2.1256,  0.9310,  1.0743],
          [ 1.9577, -0.1513,  0.1668],
          [-0.1404,  1.6647,  0.7108]],
         [[ 0.9001,  1.6930, -0.4966],
          [-1.0432, -1.0742,  1.2273],
          [-0.2711, -0.4740, -0.6381]],
         [[-1.3099, -1.7540,  0.5443],
          [ 0.3565, -2.3821,  0.8638],
          [-1.3840,  0.8216,  0.2761]],
         [[-0.5989, -0.4732,  1.3252],
          [-0.7614,  1.0493,  0.8488],
          [-0.1300,  0.1287,  0.6234]]]])
        out = torch.empty((2, 4), dtype=torch.int64)
        result = torch.linalg.matrix_rank(A, atol=1.0, rtol=0.0, hermitian=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[[[-1.1079, -0.4803,  0.2296],
          [ 0.3198, -0.2976, -0.0585],
          [-1.6931, -0.3353, -0.2893]],
         [[ 1.9757, -0.5959, -1.2041],
          [ 0.8443, -0.4916, -1.6574],
          [-0.2654, -1.0447, -0.8138]],
         [[-0.4111,  1.0973,  0.2275],
          [ 1.1851,  1.8233,  0.8187],
          [-1.4107, -0.5473,  1.1431]],
         [[ 0.0327, -0.8295,  0.0457],
          [-0.6286, -0.2507,  0.7292],
          [ 0.4075, -1.3918, -0.5015]]],
        [[[-2.1256,  0.9310,  1.0743],
          [ 1.9577, -0.1513,  0.1668],
          [-0.1404,  1.6647,  0.7108]],
         [[ 0.9001,  1.6930, -0.4966],
          [-1.0432, -1.0742,  1.2273],
          [-0.2711, -0.4740, -0.6381]],
         [[-1.3099, -1.7540,  0.5443],
          [ 0.3565, -2.3821,  0.8638],
          [-1.3840,  0.8216,  0.2761]],
         [[-0.5989, -0.4732,  1.3252],
          [-0.7614,  1.0493,  0.8488],
          [-0.1300,  0.1287,  0.6234]]]])
        out = torch.empty((2, 4), dtype=torch.int64)
        result = torch.linalg.matrix_rank(A, torch.tensor(1.), True, out=out)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)
