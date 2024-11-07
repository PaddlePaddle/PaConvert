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

obj = APIBase("torchvision.datasets.VOCDetection", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        from torchvision import transforms
        train_dataset = torchvision.datasets.VOCDetection(
            root='./data',
            image_set='trainval',
            transform=transforms.Normalize((0.5,), (0.5,)),
            download=True
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        train_dataset = paddle_aux.VOCDetection(root='./data', image_set='trainval',
            transform=paddle.vision.transforms.Normalize(mean=(0.5,), std=(0.5,)),
            download=True)
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
        train_dataset = torchvision.datasets.VOCDetection(
            root='./data',
            image_set='train',
            download=True,
            transform=transforms.Normalize((0.5,), (0.5,)),
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        train_dataset = paddle_aux.VOCDetection(root='./data', image_set='train',
            download=True, transform=paddle.vision.transforms.Normalize(mean=(0.5,),
            std=(0.5,)))
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
        train_dataset = torchvision.datasets.VOCDetection(root='./data')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        train_dataset = paddle_aux.VOCDetection(root='./data')
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
        train_dataset = torchvision.datasets.VOCDetection(root=root_path)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        root_path = './data'
        train_dataset = paddle_aux.VOCDetection(root=root_path)
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
        train = True
        train_dataset = torchvision.datasets.VOCDetection(root='./data', image_set='train')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        train = True
        train_dataset = paddle_aux.VOCDetection(root='./data', image_set='train')
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
        imageset = 'train'
        train_dataset = torchvision.datasets.VOCDetection(root='./data', image_set=imageset)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        imageset = 'train'
        train_dataset = paddle_aux.VOCDetection(root='./data', image_set=imageset)
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
        image_set_val = 'val'
        train_dataset = torchvision.datasets.VOCDetection(
            root='./data',
            image_set=image_set_val,
            transform=None,
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        image_set_val = 'val'
        train_dataset = paddle_aux.VOCDetection(root='./data', image_set=
            image_set_val, transform=None)
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        trainval_set = 'trainval'
        train_dataset = torchvision.datasets.VOCDetection('./data', image_set=trainval_set)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import sys
        sys.path.append(
            '/home/rocco/github/PaConvert-dev/tests/torchvision_tests/test_project/utils'
            )
        import paddle_aux
        import paddle
        trainval_set = 'trainval'
        train_dataset = paddle_aux.VOCDetection(root='./data', image_set=trainval_set)
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )
