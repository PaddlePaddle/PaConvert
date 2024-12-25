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


def test_case_1():
    obj = APIBase("torchvision.io.ImageReadMode.GRAY")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.io.ImageReadMode.GRAY
        """
    )
    # 这里转换后的 paddle_code 中有 `pass` 是因为转换后的代码没有使用 paddle 的 api
    # 所以 autoflake 删除了无用的 import
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "gray"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_2():
    obj = APIBase("torchvision.io.ImageReadMode.RGB")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.io.ImageReadMode.RGB
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "rgb"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_3():
    obj = APIBase("torchvision.io.ImageReadMode.UNCHANGED")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.io.ImageReadMode.UNCHANGED
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "unchanged"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_4():
    obj = APIBase("torchvision.models.AlexNet_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_5():
    obj = APIBase("torchvision.models.AlexNet_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_6():
    obj = APIBase("torchvision.models.DenseNet121_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_7():
    obj = APIBase("torchvision.models.DenseNet121_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_8():
    obj = APIBase("torchvision.models.DenseNet161_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet161_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_9():
    obj = APIBase("torchvision.models.DenseNet161_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_10():
    obj = APIBase("torchvision.models.DenseNet169_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet169_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_11():
    obj = APIBase("torchvision.models.DenseNet169_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_12():
    obj = APIBase("torchvision.models.DenseNet201_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet201_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_13():
    obj = APIBase("torchvision.models.DenseNet201_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_14():
    obj = APIBase("torchvision.models.GoogLeNet_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.GoogLeNet_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_15():
    obj = APIBase("torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_16():
    obj = APIBase("torchvision.models.Inception_V3_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_17():
    obj = APIBase("torchvision.models.Inception_V3_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_18():
    obj = APIBase("torchvision.models.MobileNet_V2_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_19():
    obj = APIBase("torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_20():
    obj = APIBase("torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_21():
    obj = APIBase("torchvision.models.MobileNet_V3_Large_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_22():
    obj = APIBase("torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_23():
    obj = APIBase("torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_24():
    obj = APIBase("torchvision.models.MobileNet_V3_Small_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_25():
    obj = APIBase("torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_26():
    obj = APIBase("torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_27():
    obj = APIBase("torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_28():
    obj = APIBase("torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_29():
    obj = APIBase("torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_30():
    obj = APIBase("torchvision.models.ResNet101_64x4d_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet101_64x4d_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_31():
    obj = APIBase("torchvision.models.ResNet101_64x4d_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet101_64x4d_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_32():
    obj = APIBase("torchvision.models.ResNet101_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet101_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_33():
    obj = APIBase("torchvision.models.ResNet101_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_34():
    obj = APIBase("torchvision.models.ResNet101_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_35():
    obj = APIBase("torchvision.models.ResNet152_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_36():
    obj = APIBase("torchvision.models.ResNet152_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_37():
    obj = APIBase("torchvision.models.ResNet152_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_38():
    obj = APIBase("torchvision.models.ResNet18_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_39():
    obj = APIBase("torchvision.models.ResNet18_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_40():
    obj = APIBase("torchvision.models.ResNet34_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_41():
    obj = APIBase("torchvision.models.ResNet34_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_42():
    obj = APIBase("torchvision.models.ResNet50_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_43():
    obj = APIBase("torchvision.models.ResNet50_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_44():
    obj = APIBase("torchvision.models.ResNet50_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_45():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_46():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_47():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_48():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_49():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_50():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_51():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_52():
    obj = APIBase("torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_53():
    obj = APIBase("torchvision.models.SqueezeNet1_0_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.SqueezeNet1_0_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_54():
    obj = APIBase("torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_55():
    obj = APIBase("torchvision.models.SqueezeNet1_1_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_56():
    obj = APIBase("torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_67():
    obj = APIBase("torchvision.models.VGG16_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG16_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_68():
    obj = APIBase("torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_FEATURES"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_69():
    obj = APIBase("torchvision.models.VGG16_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_70():
    obj = APIBase("torchvision.models.VGG19_BN_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_71():
    obj = APIBase("torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_72():
    obj = APIBase("torchvision.models.VGG19_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG19_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_73():
    obj = APIBase("torchvision.models.VGG19_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_74():
    obj = APIBase("torchvision.models.Wide_ResNet101_2_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_75():
    obj = APIBase("torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_76():
    obj = APIBase("torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_77():
    obj = APIBase("torchvision.models.Wide_ResNet50_2_Weights.DEFAULT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "DEFAULT"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_78():
    obj = APIBase("torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V1"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_79():
    obj = APIBase("torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        weights = "IMAGENET1K_V2"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_80():
    obj = APIBase("torchvision.transforms.InterpolationMode.BICUBIC")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.BICUBIC
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "bicubic"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_81():
    obj = APIBase("torchvision.transforms.InterpolationMode.BILINEAR")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.BILINEAR
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "bilinear"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_82():
    obj = APIBase("torchvision.transforms.InterpolationMode.BOX")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.BOX
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "box"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_83():
    obj = APIBase("torchvision.transforms.InterpolationMode.HAMMING")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.HAMMING
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "hamming"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_84():
    obj = APIBase("torchvision.transforms.InterpolationMode.LANCZOS")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.LANCZOS
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "lanczos"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_85():
    obj = APIBase("torchvision.transforms.InterpolationMode.NEAREST")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.NEAREST
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "nearest"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )


def test_case_86():
    obj = APIBase("torchvision.transforms.InterpolationMode.NEAREST_EXACT")
    pytorch_code = textwrap.dedent(
        """
        import torchvision
        mode = torchvision.transforms.InterpolationMode.NEAREST_EXACT
        """
    )
    paddle_code = textwrap.dedent(
        """
        pass

        mode = "nearest_exact"
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code=paddle_code,
    )
