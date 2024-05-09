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
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert isinstance(paddle_result, str)


obj = DownloadAPIBase("torch.hub.download_url_to_file")

# NOTE: Due to network limits, only test case 3
# def test_case_1():
#     pytorch_code = textwrap.dedent(
#         """
#         import torch
#         result = torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
#         """
#     )
#     obj.run(pytorch_code, ["result"], reason = "network limits, skip it")


# def test_case_2():
#     pytorch_code = textwrap.dedent(
#         """
#         import torch
#         result = torch.hub.download_url_to_file(url='https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', dst='/tmp/temporary_file')
#         """
#     )
#     obj.run(pytorch_code, ["result"], reason = "network limits, skip it")


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hub.download_url_to_file(url='https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', dst='/tmp/temporary_file',
                hash_prefix="5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8")
        """
    )
    obj.run(pytorch_code, ["result"])
