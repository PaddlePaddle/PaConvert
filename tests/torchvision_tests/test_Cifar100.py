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

obj = APIBase("torchvision.datasets.CIFAR100")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        from torchvision import transforms
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            transform=transforms.Normalize((0.5,), (0.5,)),
            download=True
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os

        import paddle

        train_dataset = paddle.vision.datasets.Cifar100(
            transform=paddle.vision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
            download=True,
            data_file=os.path.join("./data", "cifar-100-python.tar.gz"),
            mode="train",
        )
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
        from torchvision import transforms
        train_dataset = torchvision.datasets.CIFAR100('./data', True, transforms.Normalize((0.5,), (0.5,)))

        """
    )
    paddle_code = textwrap.dedent(
        """
        import os

        import paddle

        train_dataset = paddle.vision.datasets.Cifar100(
            transform=paddle.vision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
            data_file=os.path.join("./data", "cifar-100-python.tar.gz"),
            mode="train",
        )
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
        from torchvision import transforms
        train_dataset = torchvision.datasets.CIFAR100(
            train=True,
            root='./data',
            download=True,
            transform=transforms.Normalize((0.5,), (0.5,)),
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os

        import paddle

        train_dataset = paddle.vision.datasets.Cifar100(
            download=True,
            transform=paddle.vision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
            data_file=os.path.join("./data", "cifar-100-python.tar.gz"),
            mode="train",
        )
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
        train_dataset = torchvision.datasets.CIFAR100(root='./data')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os

        import paddle

        train_dataset = paddle.vision.datasets.Cifar100(
            data_file=os.path.join("./data", "cifar-100-python.tar.gz")
        )
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
        train_dataset = torchvision.datasets.CIFAR100(root=root_path, train=True)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os

        import paddle

        root_path = "./data"
        train_dataset = paddle.vision.datasets.Cifar100(
            data_file=os.path.join(root_path, "cifar-100-python.tar.gz"), mode="train"
        )
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
        train = True
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=train)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os

        import paddle

        train = True
        train_dataset = paddle.vision.datasets.Cifar100(
            data_file=os.path.join("./data", "cifar-100-python.tar.gz"),
            mode="train" if train else "test",
        )
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )
