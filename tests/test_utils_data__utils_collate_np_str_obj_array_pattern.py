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

import re
import textwrap

import pytest
from apibase import APIBase


class CollateAPIBase(APIBase):
    """APIBase with custom compare logic for collate objects (Pattern, Match)."""

    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        if isinstance(pytorch_result, re.Match):
            assert isinstance(paddle_result, re.Match), (
                f"API ({name}): paddle result should be a re.Match, "
                f"got {type(paddle_result)}"
            )
            if check_value:
                assert pytorch_result.group() == paddle_result.group(), (
                    f"API ({name}): match group mismatch, "
                    f"torch is {pytorch_result.group()}, paddle is {paddle_result.group()}"
                )
            return
        super().compare(
            name,
            pytorch_result,
            paddle_result,
            check_value,
            check_shape,
            check_dtype,
            check_stop_gradient,
            rtol,
            atol,
        )


obj = CollateAPIBase("torch.utils.data._utils.collate.np_str_obj_array_pattern")


@pytest.mark.skip(
    reason="PaConvert removes the import but does not replace the variable reference, causing NameError in Paddle code"
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data._utils.collate import np_str_obj_array_pattern
        result = np_str_obj_array_pattern
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="PaConvert converts the API to re.compile but does not add 'import re', causing NameError in Paddle code"
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.utils.data._utils.collate.np_str_obj_array_pattern
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import re
        from torch.utils.data._utils.collate import np_str_obj_array_pattern
        result = isinstance(np_str_obj_array_pattern, re.Pattern)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import re
        from torch.utils.data._utils.collate import np_str_obj_array_pattern
        result = np_str_obj_array_pattern.search("numpy_str_123")
        """
    )
    obj.run(
        pytorch_code,
        unsupport=True,
        reason="`.search()` method on this API is not supported for conversion by PaConvert",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import re
        from torch.utils.data._utils.collate import np_str_obj_array_pattern
        result = np_str_obj_array_pattern.search("object_456")
        """
    )
    obj.run(
        pytorch_code,
        unsupport=True,
        reason="`.search()` method on this API is not supported for conversion by PaConvert",
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import re
        from torch.utils.data._utils.collate import np_str_obj_array_pattern
        result = np_str_obj_array_pattern.search("tensor_789")
        """
    )
    obj.run(
        pytorch_code,
        unsupport=True,
        reason="`.search()` method on this API is not supported for conversion by PaConvert",
    )
