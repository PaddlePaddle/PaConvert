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

obj = APIBase("torch.utils.data.sampler.SubsetRandomSampler")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        sampler = SubsetRandomSampler(indices=[1, 3, 5, 7, 9])
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
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        sampler = SubsetRandomSampler([1, 3, 5, 7, 9])
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
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        sampler = SubsetRandomSampler([1, 3, 5, 7, 9], torch.Generator().manual_seed(42))
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
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        sampler = SubsetRandomSampler(indices=[1, 3, 5, 7, 9],
                                        generator=torch.Generator().manual_seed(42))
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
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        sampler = SubsetRandomSampler(generator=torch.Generator().manual_seed(42), indices=[1, 3, 5, 7, 9])
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
        import torch
        import torch.utils.data.sampler as sampler
        import numpy as np
        s = sampler.SubsetRandomSampler([2, 4, 6, 8])
        result = []
        for index in s:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    """Keyword arguments test"""
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        sampler = SubsetRandomSampler(indices=[0, 2, 4, 6, 8])
        result = []
        for index in sampler:
            result.append(index)
        result = torch.from_numpy(np.array(result).astype("float32"))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    """Mixed arguments test"""
    pytorch_code = textwrap.dedent(
        """
        from torch.utils.data.sampler import SubsetRandomSampler
        import torch
        import numpy as np
        indices = [1, 3, 5]
        sampler = SubsetRandomSampler(indices)
        result = len(sampler)
        """
    )
    obj.run(pytorch_code, ["result"])
