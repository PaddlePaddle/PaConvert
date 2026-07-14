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

obj = APIBase("torch.cuda.OutOfMemoryError")


def test_case_1():
    """Raise and catch the exception, checking the message"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        try:
            raise torch.cuda.OutOfMemoryError("test")
        except torch.cuda.OutOfMemoryError as e:
            result = str(e)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """The exception is a RuntimeError subclass"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = issubclass(torch.cuda.OutOfMemoryError, RuntimeError)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """The exception can be caught as RuntimeError"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        try:
            raise torch.cuda.OutOfMemoryError("out of memory")
        except RuntimeError as e:
            result = str(e)
        """
    )
    obj.run(pytorch_code, ["result"])
