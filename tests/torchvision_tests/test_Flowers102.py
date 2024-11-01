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

obj = APIBase("torchvision.datasets.Flowers102")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        root_path = './data'
        train_dataset = torchvision.datasets.Flowers102(root=root_path, split='train', transform=None, download=False)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.Flowers(transform=None, download=
            False, mode='train', data_file=os.path.join(root_path,
            'flowers-102/102flowers.tgz'), label_file=os.path.join(root_path,
            'flowers-102/imagelabels.mat'), setid_file=os.path.join(root_path,
            'flowers-102/setid.mat'))
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
        split = 'train'
        train_dataset = torchvision.datasets.Flowers102(root_path, split, None, download=False)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        split = 'train'
        train_dataset = paddle.vision.datasets.Flowers(download=False, mode=split if
            split != 'val' else 'valid', data_file=os.path.join(root_path,
            'flowers-102/102flowers.tgz'), label_file=os.path.join(root_path,
            'flowers-102/imagelabels.mat'), setid_file=os.path.join(root_path,
            'flowers-102/setid.mat'))
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
        train_dataset = torchvision.datasets.Flowers102(split='train', root='./data', download=False)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        train_dataset = paddle.vision.datasets.Flowers(download=False, mode='train',
            data_file=os.path.join('./data', 'flowers-102/102flowers.tgz'),
            label_file=os.path.join('./data', 'flowers-102/imagelabels.mat'),
            setid_file=os.path.join('./data', 'flowers-102/setid.mat'))
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
        train_dataset = torchvision.datasets.Flowers102(root=root_path)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.Flowers(mode='train', data_file=os.
            path.join(root_path, 'flowers-102/102flowers.tgz'), label_file=os.path.
            join(root_path, 'flowers-102/imagelabels.mat'), setid_file=os.path.join
            (root_path, 'flowers-102/setid.mat'))
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
        train_dataset = torchvision.datasets.Flowers102(root=root_path, split='test')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train_dataset = paddle.vision.datasets.Flowers(mode='test', data_file=os.
            path.join(root_path, 'flowers-102/102flowers.tgz'), label_file=os.path.
            join(root_path, 'flowers-102/imagelabels.mat'), setid_file=os.path.join
            (root_path, 'flowers-102/setid.mat'))
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
        train_dataset = torchvision.datasets.Flowers102(root=root_path, split='val')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        train = True
        train_dataset = paddle.vision.datasets.Flowers(mode='valid', data_file=os.
            path.join(root_path, 'flowers-102/102flowers.tgz'), label_file=os.path.
            join(root_path, 'flowers-102/imagelabels.mat'), setid_file=os.path.join
            (root_path, 'flowers-102/setid.mat'))
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
        split = 'val'
        download = False
        train_dataset = torchvision.datasets.Flowers102(root=root_path, split=split, download=download)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import os
        import paddle
        root_path = './data'
        split = 'val'
        download = False
        train_dataset = paddle.vision.datasets.Flowers(download=download, mode=
            split if split != 'val' else 'valid', data_file=os.path.join(root_path,
            'flowers-102/102flowers.tgz'), label_file=os.path.join(root_path,
            'flowers-102/imagelabels.mat'), setid_file=os.path.join(root_path,
            'flowers-102/setid.mat'))
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )
