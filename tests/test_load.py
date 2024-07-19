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

obj = APIBase("torch.load")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', _use_new_zipfile_serialization=False)
        result = torch.load('tensor.pt', map_location=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', _use_new_zipfile_serialization=True)
        result = torch.load('tensor.pt', map_location=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt')
        result = torch.load('tensor.pt', map_location=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', pickle_protocol=4)
        result = torch.load('tensor.pt', map_location=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


# `mmap` is not supported in paddle
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle

        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', pickle_protocol=4)
        result = torch.load('tensor.pt', map_location=torch.device('cpu'), pickle_module=pickle, weights_only=False, mmap=None)
        """
    )
    obj.run(pytorch_code, unsupport=True, reason="`mmap` is not supported in paddle")


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle

        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', pickle_protocol=4)
        result = torch.load('tensor.pt', torch.device('cpu'), pickle)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle

        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', pickle_protocol=4)
        result = torch.load('tensor.pt', weights_only=False, map_location=torch.device('cpu'), pickle_module=pickle)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch

        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', pickle_protocol=4)
        result = torch.load('tensor.pt')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from pathlib import PosixPath

        result = torch.tensor([0., 1., 2., 3., 4.])
        torch.save(result, 'tensor.pt', pickle_protocol=4)
        new_path = PosixPath('tensor.pt')
        result = torch.load(new_path)
        """
    )
    obj.run(pytorch_code, ["result"])
