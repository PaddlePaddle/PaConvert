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

obj = APIBase("torch.utils.data.WeightedRandomSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler(weights=[0.1, 0.3, 0.5, 0.7, 0.2],
                                        num_samples=5,
                                        replacement=True)
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler([0.1, 0.3, 0.5, 0.7, 0.2],
                                        5,
                                        True)
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler([0.1, 0.3, 0.5, 0.7, 0.2],
                                        5,
                                        False)
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler([0.1, 0.3, 0.5, 0.7, 0.2],
                                        10,
                                        True)
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler(num_samples=5,
                                        weights=[0.1, 0.3, 0.5, 0.7, 0.2],
                                        replacement=True)
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler([0.1, 0.3, 0.5, 0.7, 0.2], 5, True, torch.Generator().manual_seed(42))
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler(weights=[0.1, 0.3, 0.5, 0.7, 0.2],
                                        num_samples=5,
                                        replacement=True,
                                        generator=torch.Generator().manual_seed(42))
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data import WeightedRandomSampler
        import torch
        import numpy as np
        sampler = WeightedRandomSampler([0.1, 0.3, 0.5, 0.7, 0.2], 10)
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
