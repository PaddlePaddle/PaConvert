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

import paddle
from apibase import APIBase


class cudaGetDeviceProperitesAPI(APIBase):
    def compare(self, name, pytorch_result, paddle_result, check_value=True):
        return pytorch_result == paddle_result or isinstance(
            paddle_result, paddle.fluid.libpaddle._gpuDeviceProperties
        )


obj = cudaGetDeviceProperitesAPI("torch.cuda.get_device_properties")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        name = None
        major = None
        minor = None
        total_memory = None
        multi_processor_count = None
        if torch.cuda.is_available():
            result = torch.cuda.get_device_properties(0)
            name = result.name
            major = result.major
            minor = result.minor
            total_memory = result.total_memory
            multi_processor_count = result.multi_processor_count
        """
    )
    obj.run(
        pytorch_code,
        ["name", "major", "minor", "total_memory", "multi_processor_count"],
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        name = None
        major = None
        minor = None
        total_memory = None
        multi_processor_count = None
        if torch.cuda.is_available():
            result = torch.cuda.get_device_properties(device="cuda:0")
            name = result.name
            major = result.major
            minor = result.minor
            total_memory = result.total_memory
            multi_processor_count = result.multi_processor_count
        """
    )
    obj.run(
        pytorch_code,
        ["name", "major", "minor", "total_memory", "multi_processor_count"],
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        name = None
        major = None
        minor = None
        total_memory = None
        multi_processor_count = None
        if torch.cuda.is_available():
            result = torch.cuda.get_device_properties(torch.device("cuda:0"))
            name = result.name
            major = result.major
            minor = result.minor
            total_memory = result.total_memory
            multi_processor_count = result.multi_processor_count
        """
    )
    obj.run(
        pytorch_code,
        ["name", "major", "minor", "total_memory", "multi_processor_count"],
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        name = None
        major = None
        minor = None
        total_memory = None
        multi_processor_count = None
        if torch.cuda.is_available():
            t = torch.tensor([1,2,3]).cuda()
            result = torch.cuda.get_device_properties(device=torch.device("cuda:0"))
            name = result.name
            major = result.major
            minor = result.minor
            total_memory = result.total_memory
            multi_processor_count = result.multi_processor_count
        """
    )
    obj.run(
        pytorch_code,
        ["name", "major", "minor", "total_memory", "multi_processor_count"],
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        name = None
        major = None
        minor = None
        total_memory = None
        multi_processor_count = None
        if torch.cuda.is_available():
            result = torch.cuda.get_device_properties(device="cuda:0")
            name = result.name
            major = result.major
            minor = result.minor
            total_memory = result.total_memory
            multi_processor_count = result.multi_processor_count
        """
    )
    obj.run(
        pytorch_code,
        ["name", "major", "minor", "total_memory", "multi_processor_count"],
    )
