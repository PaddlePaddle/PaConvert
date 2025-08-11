# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os


class GlobalManager:
    json_file = os.path.dirname(__file__) + "/api_mapping.json"
    with open(json_file, "r") as file:
        API_MAPPING = json.load(file)

    json_file = os.path.dirname(__file__) + "/api_wildcard_mapping.json"
    with open(json_file, "r") as file:
        API_WILDCARD_MAPPING = json.load(file)

    json_file = os.path.dirname(__file__) + "/attribute_mapping.json"
    with open(json_file, "r") as file:
        ATTRIBUTE_MAPPING = json.load(file)

    json_file = os.path.dirname(__file__) + "/api_alias_mapping.json"
    with open(json_file, "r") as file:
        ALIAS_MAPPING = json.load(file)

    json_file = os.path.dirname(__file__) + "/api_alias_mapping.json"
    with open(json_file, "r") as file:
        ALIAS_MAPPING = json.load(file)

    # used to replace import (means replace api by all)
    IMPORT_PACKAGE_MAPPING = {
        "audiotools": "paddlespeech.audiotools",
    }
    # used to replace api one by one
    # Abbreviation after annotation as the prefix for corresponding matcher
    TORCH_PACKAGE_MAPPING = {
        "torch": "paddle",
        "mmseg": "paddle",
        "mmcv": "paddle",
        "mmdet": "paddle",
        "mmdet3d": "paddle",
        "mmengine": "paddle",
        "detectron": "paddle",
        "timm": "paddle",
        "torchvision": "paddle",
        "torchaudio": "paddlespeech",
        "kornia": "paddle",
        "fasttext": "paddle",
        "pytorch_lightning": "paddle",
        "lightning": "paddle",
        "jieba": "paddle",
        "NLTK": "paddle",
        "scikit-learn": "paddle",
        "fairscale": "paddle",  # FS
        "transformers": "paddlenlp",  # TRFM
        "datasets": "paddle",
        "accelerate": "paddle",
        "diffusers": "paddle",
        "torch_xla": "paddle",
        "flash_attn": "paddle",  # FA
    }
    MAY_TORCH_PACKAGE_LIST = [
        "setuptools",
        "os",
        "einops",
    ]

    # 无需转换的Pytorch API名单
    NO_NEED_CONVERT_LIST = [
        "torch.einsum",
        "torch.Tensor.cos",
        "torch.Tensor.masked_scatter",
        "torch.nn.functional.dropout1d",
    ]
