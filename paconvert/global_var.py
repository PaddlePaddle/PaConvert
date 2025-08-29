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

        # zhimin
        "torch.Tensor.bfloat16",
        "torch.Tensor.bool",
        "torch.Tensor.byte",
        "torch.Tensor.char",
        "torch.Tensor.double",
        "torch.Tensor.float",
        "torch.Tensor.half",
        "torch.Tensor.int",
        "torch.Tensor.long",
        "torch.Tensor.short",
        "torch.Tensor.cfloat",
        "torch.Tensor.cdouble",
        "torch.nn.init.calculate_gain",
        "torch.nn.init.constant_",
        "torch.nn.init.dirac_",
        "torch.nn.init.eye_",
        "torch.nn.init.kaiming_normal_",
        "torch.nn.init.kaiming_uniform_",
        "torch.nn.init.normal_",
        "torch.nn.init.ones",
        "torch.nn.init.orthogonal_",
        "torch.nn.init.trunc_normal_",
        "torch.nn.init.uniform_",
        "torch.nn.init.xavier_normal_",
        "torch.nn.init.xavier_uniform_",
        "torch.nn.init.zeros_",
        "torch.nn.Conv1d",
        "torch.nn.Conv2d",
        "torch.nn.Conv3d",
        "torch.nn.Embedding",

        # zhouxin
        "torch.view_as_real",
        "torch.view_as_complex",
        "torch.autograd.Function",
        "torch.argwhere",


        # honggeng
        "torch.nn.functional.dropout1d",
        "torch.nn.parameter.Parameter",
        "torch.add",
        "torch.div",
        "torch.divide",
        "torch.true_divide",
        "torch.Tensor.add",
        "torch.Tensor.add_",
        "torch.Tensor.div",
        "torch.Tensor.div_",
        "torch.Tensor.divide",
        "torch.Tensor.divide_",
        "torch.Tensor.true_divide",


        # sensen


        # hongyu


        # linjun
        "torch.as_tensor", 
        "torch.tensor",
        "torch.Tensor.copy_",
        "torch.Tensor.norm",
        # "torch.Tensor",
        "torch.FloatTensor", 
        "torch.DoubleTensor",
        "torch.HalfTensor",
        "torch.BFloat16Tensor",
        "torch.ByteTensor",
        "torch.CharTensor",
        "torch.ShortTensor",
        "torch.IntTensor",
        "torch.LongTensor",
        "torch.BoolTensor",


        # siyu


        # shijie
        "torch.msort",
        "torch.Tensor.msort",
        "torch.Tensor.ravel",
        "torch.ravel",
        "torch.Tensor.scatter_add",
        "torch.scatter_add",
        "torch.Tensor.scatter_add_",
        "torch.Tensor.tril",
        "torch.tril",
        "torch.Tensor.triu",
        "torch.triu",
        "torch.bmm",
        "torch.Tensor.bmm",
        "torch.nn.GELU",
        "torch.broadcast_shapes",
        "torch.Tensor.scatter_reduce",
        "torch.scatter_reduce",


        # yuyan


        # huoda
        "torch.get_default_dtype",
        "torch.einsum",
        "torch.nn.Identity",
        # "torch.nn.MSELoss",
        # "torch.nn.CrossEntropyLoss",
        # "torch.nn.BCEWithLogitsLoss",
        # "torch.nn.functional.cross_entropy",
        "torch.Tensor.device",
        "torch.Tensor.ndim",
        "torch.Tensor.T",
        "torch.Tensor.abs",
        "torch.Tensor.cos",
        "torch.Tensor.detach",
        "torch.Tensor.dim",
        "torch.Tensor.fill_",
        "torch.Tensor.isnan",
        "torch.Tensor.item",
        "torch.Tensor.log",
        "torch.Tensor.masked_scatter",
        "torch.Tensor.masked_fill_",
        "torch.Tensor.masked_fill",
        "torch.Tensor.nonzero",
        "torch.Tensor.normal_",
        "torch.Tensor.numel",
        "torch.Tensor.sigmoid",
        "torch.Tensor.sin",
        "torch.Tensor.square",
        "torch.Tensor.tolist",
        "torch.Tensor.zero_",
        "torch.distributed.get_rank",
        "torch.distributed.get_world_size",
        "torch.special.softmax",
        "torch.Tensor.shape",
        "torch.float32",
        "torch.long",
        "torch.int32",
        "torch.bfloat16",
        "torch.int64",
        "torch.bool",
        "torch.uint8",


        # sundong
        "torch.amax"
        "torch.amin"
        "torch.Tensor.amax"
        "torch.Tensor.amin"
        
        # zhengsheng


        # liuyi
        "torch.finfo",
        "torch.is_complex",
        "torch.concat",
        "torch.empty_like",
        "torch.full",
        "torch.nonzero",
        "torch.Tensor.pow",
        "torch.Tensor.prod",
        "torch.Tensor.reshape",
        "torch.zeros_like",
        "torch.argsort",
        "torch.Tensor.argsort",
        "torch.Tensor.squeeze",
        "torch.chunk",
        "torch.Tensor.chunk",
        "torch.any",

        # shenwei
        "torch.Tensor.expand_as",
        "torch.logsumexp",
        "torch.Tensor.logsumexp",
        "torch.argmax",
        "torch.Tensor.argmax",
        "torch.argmin",
        "torch.Tensor.argmin",
        "torch.all",
        "torch.Tensor.all",

        # haoyang
        "torch.logical_not",
        "torch.Tensor.logical_not",
        "torch.logical_and",
        "torch.Tensor.logical_and",
        "torch.logical_or",
        "torch.Tensor.logical_or",
        "torch.logical_xor",
        "torch.Tensor.logical_xor",
        "torch.index_select",
        "torch.Tensor.index_select",


        # rongrui


        # bingxin
        
    ]
