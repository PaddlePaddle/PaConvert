# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#

import textwrap

from apibase import APIBase

obj1 = APIBase("torch.utils.data._utils.collate.np_str_obj_array_pattern")
obj2 = APIBase("torch.utils.data._utils.collate.default_collate_err_msg_format")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import re
        from torch.utils.data._utils.collate import np_str_obj_array_pattern
        result = np_str_obj_array_pattern
        """
    )
    obj1.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data._utils.collate import default_collate_err_msg_format
        result = default_collate_err_msg_format
        """
    )
    obj2.run(pytorch_code, ["result"])


# TODO: unsupported when *.search/*.format
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
        if np_str_obj_array_pattern.search('test') is not None:
            raise TypeError(default_collate_err_msg_format.format(elem.dtype))
        """
    )
    obj2.run(pytorch_code, unsupport=True, reason="TODO unsupported")
