# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import pytest
from apibase import APIBase

obj = APIBase("torchvision.io.decode_jpeg")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.io as io
        from torchvision.io import ImageReadMode
        import cv2
        import numpy as np
        fake_img = np.ones((400, 300, 3), dtype='uint8') * 255
        cv2.imwrite('fake.jpg', fake_img)
        img_bytes = io.read_file('fake.jpg')
        result = io.decode_jpeg(input=img_bytes, mode=ImageReadMode.RGB, device=torch.device('cuda'))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.io as io
        from torchvision.io import ImageReadMode
        import cv2
        import numpy as np
        fake_img = np.zeros((400, 300, 3), dtype='uint8')
        cv2.imwrite('fake.jpg', fake_img)
        img_bytes = io.read_file('fake.jpg')
        result = io.decode_jpeg(img_bytes, ImageReadMode.GRAY, torch.device('cuda'))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.io as io
        from torchvision.io import ImageReadMode
        import cv2
        import numpy as np
        fake_img = np.ones((400, 300, 3), dtype='uint8') * 128
        cv2.imwrite('fake.jpg', fake_img)
        img_bytes = io.read_file('fake.jpg')
        result = io.decode_jpeg(img_bytes, device=torch.device('cuda'), mode=ImageReadMode.UNCHANGED)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.io as io
        import cv2
        import numpy as np
        fake_img = np.full((400, 300, 3), 200, dtype='uint8')
        cv2.imwrite('fake.jpg', fake_img)
        img_bytes = io.read_file('fake.jpg')
        result = io.decode_jpeg(img_bytes, device=torch.device('cuda'))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision.io as io
        from torchvision.io import ImageReadMode
        import cv2
        import numpy as np
        fake_img = np.zeros((400, 300, 3), dtype='uint8')
        fake_img[::2, ::2] = 255
        cv2.imwrite('fake.jpg', fake_img)
        img_bytes = io.read_file('fake.jpg')
        result = io.decode_jpeg(input=img_bytes, mode=ImageReadMode.RGB, device=torch.device('cuda'))
        """
    )
    obj.run(pytorch_code, ["result"])
