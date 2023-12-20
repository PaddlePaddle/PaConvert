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

import numpy as np
from apibase import APIBase


class MaxPoolAPI(APIBase):
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
        if isinstance(pytorch_result, (tuple, list)):
            for i in range(len(pytorch_result)):
                self.compare(self.pytorch_api, pytorch_result[i], paddle_result[i])
            return

        pytorch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy()
        assert (
            pytorch_result.requires_grad != paddle_result.stop_gradient
        ), "API ({}): requires grad mismatch, torch tensor's requires_grad is {}, paddle tensor's stop_gradient is {}".format(
            name, pytorch_result.requires_grad, paddle_result.stop_gradient
        )
        assert (
            pytorch_numpy.shape == paddle_numpy.shape
        ), "API ({}): shape mismatch, torch shape is {}, paddle shape is {}".format(
            name, pytorch_numpy.shape, paddle_numpy.shape
        )
        assert np.allclose(
            pytorch_numpy, paddle_numpy
        ), "API ({}): paddle result has diff with pytorch result".format(name)


obj = MaxPoolAPI("torch.nn.AdaptiveMaxPool2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(5)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(output_size=(2, 2))
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(output_size=2)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(output_size=(None, 2))
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d((None, 2), True)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(output_size=(None, 2), return_indices=True)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_6
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                            [-0.0832, -0.5456, -0.5009,  1.5103],
                            [-1.2860,  1.0287, -1.3902,  0.4627],
                            [-0.0502, -1.3924, -0.3327,  0.1678]]]])
        model = nn.AdaptiveMaxPool2d(return_indices=True, output_size=(None, 2))
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])
