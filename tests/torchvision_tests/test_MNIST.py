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

from apibase import APIBase

obj = APIBase("torchvision.datasets.MNIST")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train_dataset = torchvision.datasets.MNIST(root=root_path, train=True, transform=None, download=False)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.MNIST(transform=None, download=False,
            mode='train', image_path=os.path.join(root_path,
            'MNIST/raw/train-images-idx3-ubyte.gz'), label_path=os.path.join(
            root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train_dataset = torchvision.datasets.MNIST(root_path, True, None, download=False)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.MNIST(transform=None, download=False,
            mode='train', image_path=os.path.join(root_path,
            'MNIST/raw/train-images-idx3-ubyte.gz'), label_path=os.path.join(
            root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train_dataset = torchvision.datasets.MNIST(train=True, root=root_path, download=False)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.MNIST(download=False, mode='train',
            image_path=os.path.join(root_path,
            'MNIST/raw/train-images-idx3-ubyte.gz'), label_path=os.path.join(
            root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train_dataset = torchvision.datasets.MNIST(root=root_path)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.MNIST(mode='train', image_path=os.
            path.join(root_path, 'MNIST/raw/train-images-idx3-ubyte.gz'),
            label_path=os.path.join(root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train_dataset = torchvision.datasets.MNIST(root=root_path, train=True)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.MNIST(mode='train', image_path=os.
            path.join(root_path, 'MNIST/raw/train-images-idx3-ubyte.gz'),
            label_path=os.path.join(root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train = True
        train_dataset = torchvision.datasets.MNIST(root=root_path, train=train)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train = True
        train_dataset = paddle.vision.datasets.MNIST(mode='train' if train else
            'test', image_path=os.path.join(root_path,
            'MNIST/raw/train-images-idx3-ubyte.gz') if train else os.path.join(
            root_path, 'MNIST/raw/t10k-images-idx3-ubyte.gz'), label_path=os.path.
            join(root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz') if train else
            os.path.join(root_path, 'MNIST/raw/t10k-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train = True
        download = False
        train_dataset = torchvision.datasets.MNIST(root=root_path, train=train, download=download)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train = True
        download = False
        train_dataset = paddle.vision.datasets.MNIST(download=download, mode=
            'train' if train else 'test', image_path=os.path.join(root_path,
            'MNIST/raw/train-images-idx3-ubyte.gz') if train else os.path.join(
            root_path, 'MNIST/raw/t10k-images-idx3-ubyte.gz'), label_path=os.path.
            join(root_path, 'MNIST/raw/train-labels-idx1-ubyte.gz') if train else
            os.path.join(root_path, 'MNIST/raw/t10k-labels-idx1-ubyte.gz'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )
