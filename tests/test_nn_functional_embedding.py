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

obj = APIBase("torch.nn.functional.embedding")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(x,embedding_matrix)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(x,embedding_matrix,padding_idx=0)
        """
    )

    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(x, embedding_matrix, padding_idx=0, max_norm=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                           [1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.]])
        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(input=x, weight=embedding_matrix, padding_idx=0, max_norm=2, norm_type=2.0, scale_grad_by_freq=True, sparse=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport scale_grad_by_freq ",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                           [1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.]])
        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(input=x, padding_idx=0, max_norm=2, weight=embedding_matrix, scale_grad_by_freq=True, norm_type=2.0, sparse=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport scale_grad_by_freq ",
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                           [1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.]])
        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(x, embedding_matrix, 0, 2, 2.0, True, True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport scale_grad_by_freq ",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        embedding_matrix = torch.Tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

        x = torch.tensor(np.array([[0,1],[2,3]]))
        result = torch.nn.functional.embedding(x, embedding_matrix, padding_idx=0, max_norm=2, norm_type=1.5)
        """
    )
    obj.run(pytorch_code, ["result"])
