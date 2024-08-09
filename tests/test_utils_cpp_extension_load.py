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

obj = APIBase("torch.utils.cpp_extension.load")


# need to add cpp file
def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import load
        result = load(name='extension', sources=['extension.cpp'])
        """
    )
    obj.run(pytorch_code, ["extension"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import load
        extension = load(
            name='my_cpp_extension',
            sources=['extension.cpp']
            extra_cflags=['-fPIC'],
            extra_ldflags=['-L/path/to/libs', '-lmylib']
        )
        """
    )
    obj.run(pytorch_code, ["extension"])


def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import load
        extension = load(
            name='my_cpp_extension',
            sources=['path/to/source.cpp', 'path/to/header.h']
        )
        """
    )
    obj.run(pytorch_code, ["extension"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import load
        extension = load(
            name='my_cpp_extension',
            sources=['path/to/source.cpp'],
            extra_cflags=['-O3'],
            extra_ldflags=['-lmylib']
        )
        """
    )
    obj.run(pytorch_code, ["extension"])


def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.cpp_extension import load
        extension = load(
            name='my_cpp_extension',
            extra_cflags=['-fPIC'],
            extra_ldflags=['-L/absolute/path/to/libs', '-lyourlib'],
            s_python_module= Falsse
        )
        """
    )
    obj.run(pytorch_code, ["extension"])
