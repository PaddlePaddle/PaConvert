# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.distributed.ProcessGroup")


def test_case_1():
    """Use as a type annotation."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.distributed import ProcessGroup
        def f(pg: ProcessGroup):
            return pg
        """
    )
    expect = textwrap.dedent(
        """
        import paddle


        def f(pg: paddle.distributed.ProcessGroup):
            return pg
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)


def test_case_2():
    """Fully qualified attribute access."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        cls = torch.distributed.ProcessGroup
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        cls = paddle.distributed.ProcessGroup
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)
