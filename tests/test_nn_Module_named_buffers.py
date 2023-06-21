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

obj = APIBase("torch.nn.Module.named_buffers")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SubModel(nn.Module):
            def __init__(self):
                super(SubModel, self).__init__()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                return x

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.sub = SubModel()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                return x

        model = Model()
        result = []
        for name, buf in model.named_buffers():
            result.append(buf)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SubModel(torch.nn.Module):
            def __init__(self):
                super(SubModel, self).__init__()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.sub = SubModel()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass

        model = Model()
        result = []
        for name, buf in model.named_buffers():
            result.append(name)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SubModel(torch.nn.Module):
            def __init__(self):
                super(SubModel, self).__init__()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.sub = SubModel()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass


        model = Model()
        result = []
        for name, buf in model.named_buffers(prefix='wfs'):
            result.append(name)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SubModel(torch.nn.Module):
            def __init__(self):
                super(SubModel, self).__init__()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.sub = SubModel()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass


        model = Model()
        result = []
        for name, buf in model.named_buffers(recurse=False):
            result.append(buf)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class SubModel(torch.nn.Module):
            def __init__(self):
                super(SubModel, self).__init__()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.sub = SubModel()
                self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
                self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

            def forward(self, x):
                pass


        model = Model()
        result = []
        for name, buf in model.named_buffers(recurse=False):
            result.append(name)
        """
    )
    obj.run(pytorch_code, ["result"])
