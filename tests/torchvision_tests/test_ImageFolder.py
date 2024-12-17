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
#
import textwrap

from apibase import APIBase

obj = APIBase("torchvision.datasets.ImageFolder")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        from pathlib import Path
        import torchvision
        fake_data_dir = './text_data'
        image_folder = torchvision.datasets.ImageFolder(Path(fake_data_dir))
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        from pathlib import Path
        fake_data_dir = './text_data'
        image_folder = paddle.vision.datasets.ImageFolder(root=Path(fake_data_dir))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        from pathlib import Path
        import torchvision
        import cv2
        fake_data_dir = './text_data'
        image_folder = torchvision.datasets.ImageFolder(
            fake_data_dir,
            loader=lambda x: cv2.imread(x),
            transform=transform,
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        from pathlib import Path
        import cv2
        fake_data_dir = './text_data'
        image_folder = paddle.vision.datasets.ImageFolder(root=fake_data_dir,
            loader=lambda x: cv2.imread(x), transform=transform)
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        from pathlib import Path
        import torchvision
        import cv2
        fake_data_dir = './text_data'
        image_folder = torchvision.datasets.ImageFolder(
            root=fake_data_dir,
            transform=transform,
            loader=lambda x: cv2.imread(x),
            is_valid_file=lambda x: x.endswith('.jpg'),
            allow_empty=True
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        from pathlib import Path
        import cv2
        fake_data_dir = './text_data'
        image_folder = paddle.vision.datasets.ImageFolder(root=fake_data_dir,
            transform=transform, loader=lambda x: cv2.imread(x), is_valid_file=lambda
            x: x.endswith('.jpg'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        from pathlib import Path
        import torchvision
        import cv2
        fake_data_dir = './text_data'
        image_folder = torchvision.datasets.ImageFolder(
            fake_data_dir,
            transform,
            loader = lambda x: cv2.imread(x),
            is_valid_file = lambda x: x.endswith('.jpg'),
            allow_empty = True
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        from pathlib import Path
        import cv2
        fake_data_dir = './text_data'
        image_folder = paddle.vision.datasets.ImageFolder(root=fake_data_dir,
            transform=transform, loader=lambda x: cv2.imread(x), is_valid_file=lambda
            x: x.endswith('.jpg'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        from pathlib import Path
        import torchvision
        import cv2
        fake_data_dir = './text_data'
        image_folder = torchvision.datasets.ImageFolder(
            loader=lambda x: cv2.imread(x),
            root=fake_data_dir,
            allow_empty=True,
            transform=transform,
            is_valid_file=lambda x: x.endswith('.jpg'),
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        from pathlib import Path
        import cv2
        fake_data_dir = './text_data'
        image_folder = paddle.vision.datasets.ImageFolder(loader=lambda x: cv2.
            imread(x), root=fake_data_dir, transform=transform, is_valid_file=lambda
            x: x.endswith('.jpg'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        from pathlib import Path
        import torchvision
        fake_data_dir = './text_data'
        image_folder = torchvision.datasets.ImageFolder(
            root=fake_data_dir
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle
        from pathlib import Path
        fake_data_dir = './text_data'
        image_folder = paddle.vision.datasets.ImageFolder(root=fake_data_dir)
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )
