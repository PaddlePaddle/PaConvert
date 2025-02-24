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
#

import textwrap

from apibase import APIBase


class DownloadAPIBase(APIBase):
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
        assert isinstance(paddle_result, str)


obj = DownloadAPIBase("torch.hub.download_url_to_file")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file('https://paddle-paconvert.bj.bcebos.com/model.params', '/tmp/temporary_file')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file(url='https://paddle-paconvert.bj.bcebos.com/model.params', dst='/tmp/temporary_file')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file(url='https://paddle-paconvert.bj.bcebos.com/model.params', dst='/tmp/temporary_file',
                hash_prefix="e1bf0a03102811bb2168e9952fe4edfa09cceb3343278bd4e5876b33b6889e9b")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file(url='https://paddle-paconvert.bj.bcebos.com/model.params', dst='/tmp/temporary_file',
                hash_prefix="e1bf0a03102811bb2168e9952fe4edfa09cceb3343278bd4e5876b33b6889e9b", progress=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file(url='https://paddle-paconvert.bj.bcebos.com/model.params', dst='/tmp/temporary_file',
                hash_prefix="e1bf0a03102811bb2168e9952fe4edfa09cceb3343278bd4e5876b33b6889e9b", progress=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file(dst='/tmp/temporary_file',
                hash_prefix="e1bf0a03102811bb2168e9952fe4edfa09cceb3343278bd4e5876b33b6889e9b", url='https://paddle-paconvert.bj.bcebos.com/model.params', progress=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file('https://paddle-paconvert.bj.bcebos.com/model.params', '/tmp/temporary_file',
                "e1bf0a03102811bb2168e9952fe4edfa09cceb3343278bd4e5876b33b6889e9b", False)
        """
    )
    obj.run(pytorch_code, ["result"])
