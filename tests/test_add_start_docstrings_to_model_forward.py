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

obj = APIBase("transformers.utils.add_start_docstrings_to_model_forward")


# can only be run in file, cannot be run in exec
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch import nn
        from transformers.utils import add_start_docstrings_to_model_forward
        class LlamaForCausalLM(nn.Module):
            @add_start_docstrings_to_model_forward('test docstring')
            def forward(self, input_ids):
                return input_ids
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        import paddleformers


        class LlamaForCausalLM(paddle.nn.Module):
            @paddleformers.trainer.utils.add_start_docstrings_to_model_forward("test docstring")
            def forward(self, input_ids):
                return input_ids
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)
