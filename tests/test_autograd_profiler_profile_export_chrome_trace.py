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

obj = APIBase("torch.autograd.profiler.profile.export_chrome_trace")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import tempfile
        from torch.autograd import profiler
        def test_function():
            pass

        def get_temp_filename():
            temp_file = tempfile.NamedTemporaryFile()
            temp_filename = temp_file.name
            temp_file.close()
            return temp_filename

        with profiler.profile(record_shapes=True) as prof:
            test_function()

        temp_filename = get_temp_filename()
        prof.export_chrome_trace(temp_filename)
        result = None
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="The transformation for the 'torch.autograd.profiler.profile' is currently not supported",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import tempfile
        from torch.autograd import profiler
        def test_function():
            pass

        def get_temp_filename():
            temp_file = tempfile.NamedTemporaryFile()
            temp_filename = temp_file.name
            temp_file.close()
            return temp_filename

        with profiler.profile(record_shapes=True) as prof:
            test_function()

        temp_filename = get_temp_filename()
        prof.export_chrome_trace(path=temp_filename)
        result = None
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="The transformation for the 'torch.autograd.profiler.profile' is currently not supported",
    )
