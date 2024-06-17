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

obj = APIBase("torch.nn.Embedding", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        embedding = torch.nn.Embedding(4, 3)
        w0 = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
        with torch.no_grad():
            embedding.weight[0]=w0[0]
            embedding.weight[1]=w0[1]
            embedding.weight[3]=w0[3]
        x = torch.LongTensor([[0],[1],[3]])
        result = embedding(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        padding_idx = 0
        embedding = torch.nn.Embedding(4, 3,padding_idx=padding_idx)
        w0 = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
        with torch.no_grad():
            embedding.weight[0]=w0[0]
            embedding.weight[1]=w0[1]
            embedding.weight[2]=w0[2]
            embedding.weight[3]=w0[3]
        x = torch.LongTensor([[0],[1],[3]])
        result = embedding(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        padding_idx = 0
        embedding = torch.nn.Embedding(4, 3,padding_idx=padding_idx,max_norm=2.0)
        w0 = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
        with torch.no_grad():
            embedding.weight[0]=w0[0]
            embedding.weight[1]=w0[1]
            embedding.weight[2]=w0[2]
            embedding.weight[3]=w0[3]
        x = torch.LongTensor([[0],[1],[3]])
        result = embedding(x)
        """
    )
    obj.run(pytorch_code, unsupport=True, reason="paddle unsupport")


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        padding_idx = 0
        embedding = torch.nn.Embedding(4, 3,padding_idx=padding_idx,max_norm=2.0)
        result = embedding.padding_idx
        """
    )
    obj.run(pytorch_code)
