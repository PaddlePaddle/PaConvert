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

obj = APIBase("torch.Tensor.rename")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        x.rename(columns={'iids': iids})
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code='import paddle\n\nx = paddle.to_tensor(data=[1, 2, 3])\nx.rename(columns={"iids": iids})\n',
    )


# Fine, convert is not supported
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
        renamed_imgs = imgs.rename(N='batch', C='channels')
        esult = renamed_imgs.names
        """
    )
    obj.run(pytorch_code, ["result"])
