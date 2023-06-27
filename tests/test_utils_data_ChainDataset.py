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

obj = APIBase("torch.utils.data.ChainDataset")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import math
        import torch
        from torch.utils.data import IterableDataset, ChainDataset
        class MyIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, start, end):
                super(MyIterableDataset).__init__()
                assert end > start, "this example code only works with end >= start"
                self.start = start
                self.end = end

            def __iter__(self):
                iter_start = self.start
                iter_end = self.end
                return iter(range(iter_start, iter_end))


        dataset = ChainDataset([MyIterableDataset(start=3, end=7), MyIterableDataset(start=3, end=7)])
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import math
        import torch
        from torch.utils.data import IterableDataset, ChainDataset
        class MyIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, start, end):
                super(MyIterableDataset).__init__()
                assert end > start, "this example code only works with end >= start"
                self.start = start
                self.end = end

            def __iter__(self):
                iter_start = self.start
                iter_end = self.end
                return iter(range(iter_start, iter_end))


        dataset = ChainDataset([MyIterableDataset(start=1, end=10), MyIterableDataset(start=1, end=3)])
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import math
        import torch
        from torch.utils.data import IterableDataset, ChainDataset
        class MyIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, start, end):
                super(MyIterableDataset).__init__()
                assert end > start, "this example code only works with end >= start"
                self.start = start
                self.end = end

            def __iter__(self):
                iter_start = self.start
                iter_end = self.end
                return iter(range(iter_start, iter_end))


        dataset = ChainDataset([MyIterableDataset(start=1, end=10)])
        result = []
        for d in dataset:
            result.append(d)
        """
    )
    obj.run(pytorch_code, ["result"])
