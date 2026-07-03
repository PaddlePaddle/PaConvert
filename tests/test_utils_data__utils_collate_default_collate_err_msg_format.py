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

obj = APIBase("torch.utils.data._utils.collate.default_collate_err_msg_format")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data._utils.collate import default_collate_err_msg_format
        result = default_collate_err_msg_format
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.utils.data._utils.collate.default_collate_err_msg_format
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data._utils.collate import default_collate_err_msg_format
        result = isinstance(default_collate_err_msg_format, str)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.utils.data._utils.collate as collate
        result = collate.default_collate_err_msg_format
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.utils.data._utils.collate.default_collate_err_msg_format
        result_check = "default_collate" in result
        """
    )
    obj.run(pytorch_code, ["result_check"])
