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

test_cases = {
    "torchvision.io.ImageReadMode.GRAY": "gray",
    "torchvision.io.ImageReadMode.RGB": "rgb",
    "torchvision.io.ImageReadMode.UNCHANGED": "unchanged",
    "torchvision.models.AlexNet_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.AlexNet_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.DenseNet121_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.DenseNet121_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.DenseNet161_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.DenseNet161_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.DenseNet169_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.DenseNet169_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.DenseNet201_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.DenseNet201_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.GoogLeNet_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.Inception_V3_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.Inception_V3_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.MobileNet_V2_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.MobileNet_V3_Large_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.MobileNet_V3_Small_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.ResNet101_64x4d_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNet101_64x4d_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNet101_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNet101_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNet101_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.ResNet152_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNet152_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNet152_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.ResNet18_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNet18_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNet34_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNet34_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNet50_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ResNet50_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ResNet50_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.SqueezeNet1_0_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.SqueezeNet1_1_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG11_BN_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG11_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG11_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG13_BN_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG13_BN_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG13_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG13_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG16_BN_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG16_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES": "IMAGENET1K_FEATURES",
    "torchvision.models.VGG16_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG19_BN_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.VGG19_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.VGG19_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.Wide_ResNet101_2_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.models.Wide_ResNet50_2_Weights.DEFAULT": "DEFAULT",
    "torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1": "IMAGENET1K_V1",
    "torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2": "IMAGENET1K_V2",
    "torchvision.transforms.InterpolationMode.BICUBIC": "bicubic",
    "torchvision.transforms.InterpolationMode.BILINEAR": "bilinear",
    "torchvision.transforms.InterpolationMode.BOX": "box",
    "torchvision.transforms.InterpolationMode.HAMMING": "hamming",
    "torchvision.transforms.InterpolationMode.LANCZOS": "lanczos",
    "torchvision.transforms.InterpolationMode.NEAREST": "nearest",
    "torchvision.transforms.InterpolationMode.NEAREST_EXACT": "nearest_exact",
}


def create_test_case(pytorch_enum, paddle_value):
    def test_case():
        obj = APIBase(pytorch_enum)
        pytorch_code = textwrap.dedent(
            f"""
            import torchvision
            mode = {pytorch_enum}
        """
        )
        paddle_code = textwrap.dedent(
            f"""
            pass

            mode = "{paddle_value}"
        """
        )
        obj.run(
            pytorch_code,
            expect_paddle_code=paddle_code,
        )

    return test_case


for pytorch_enum, paddle_value in test_cases.items():
    test_func = create_test_case(pytorch_enum, paddle_value)
    test_func.__name__ = f"test_case_{pytorch_enum.replace('.', '_').replace('/', '_')}"
    globals()[test_func.__name__] = test_func
