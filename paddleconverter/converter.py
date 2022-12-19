# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import os
import logging
import ast
import astor
import shutil
import collections


from .transformer.import_transformer import ImportTransformer
from .transformer.basic_transformer import BasicTransformer

class Converter:
    def __init__(self, log_dir=None):
        self.imports_map = collections.defaultdict(dict)
        self.torch_api_count = 0
        self.success_api_count = 0
        if log_dir is None:
            self.log_dir = os.getcwd()+ '/convert.log'
        else:
            self.log_dir = log_dir
        self.logger = logging.getLogger(name='Converter')
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(self.log_dir))
        self.logger.setLevel(logging.INFO)

        self.log_info("======================================")
        self.log_info("PyTorch to Paddle Convert Start----->:")
        self.log_info("======================================")


    def run(self, in_dir, out_dir):
        in_dir = os.path.abspath(in_dir)
        out_dir = os.path.abspath(out_dir)

        if os.path.isfile(in_dir):
            old_path = in_dir
            
            # out is directory name
            if '.' not in out_dir:
                new_path = os.path.join(out_dir, os.path.basename(old_path))
            # out is file name
            else:
                new_path = out_dir
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            self.transfer_file(old_path, new_path)
        elif os.path.isdir(in_dir):
            in_dir_items = os.listdir(in_dir)
            for item in in_dir_items:
                old_path = os.path.join(in_dir, item)
                new_path = os.path.join(out_dir, item)
                if os.path.isdir(old_path):
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                self.run(old_path, new_path)
        else:
            raise ValueError(" the input 'in_dir' must be a file or directory! ")

        faild_api_count = self.torch_api_count - self.success_api_count
        self.log_info("\n======================================")
        self.log_info("Convert Summary:")
        self.log_info("======================================")
        self.log_info("There are {} Pytorch APIs in this Project:".format(self.torch_api_count))
        self.log_info(" {}  Pytorch APIs have been converted to Paddle successfully!".format(self.success_api_count))
        self.log_info(" {}  Pytorch APIs are converted failed!".format(faild_api_count))
        self.log_info(" Convert Rate is: {:.2%}".format(self.success_api_count/self.torch_api_count))
        if (faild_api_count > 0):
            self.log_info("\nFor these {} converted failed Pytorch APIs, Please refer to "
        "https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#pytorch-1-8-paddle-2-0-api "
        "and modify it by yourself manually!".format(faild_api_count))
        self.log_info("\nThank you to use Paddle Convert tool. You can make any suggestions to us.\n")
        return self.success_api_count, faild_api_count

    def mark_unsport(self, code):
        lines = code.split('\n')
        mark_next_line = False
        for i, line in enumerate(lines):
            if 'torch' in line:
                lines[i] = ">>> " + line
            if 'Torch Tensor Method' in line:
                mark_next_line = True
            else:
                if mark_next_line:
                    lines[i] = ">>> " + line
                    mark_next_line = False

        return '\n'.join(lines)


    def log_info(self, msg, file=None, line=None):
        if file:
            if line:
                msg = "[{}:{}] {}".format(file, line, msg)
            else:
                msg = "[{}] {}".format(file, msg)
        else:
            msg = "{}".format(msg)
        self.logger.info(msg)


    def transfer_file(self, old_path, new_path):
        if old_path.endswith(".py"):
            self.log_info("Start convert {} ---> {}".format(old_path, new_path))
            with open(old_path, 'r') as f:
                code = f.read()
                root = ast.parse(code)
            
            self.transfer_node(root, old_path)
            code = astor.to_source(root)
            code = self.mark_unsport(code)

            with open(new_path, 'w') as file:
                file.write(code)
            self.log_info("Finish convert {} ---> {}".format(old_path, new_path))
        elif old_path.endswith("requirements.txt"):
            self.log_info("Start convert {} ---> {}".format(old_path, new_path))
            with open(old_path, 'r') as old_file:
                code = old_file.read()
            code = code.replace('torch', 'paddlepaddle')
            with open(new_path, 'w') as new_file:
                new_file.write(code)
            self.log_info("Finish convert {} ---> {}".format(old_path, new_path))
        else:
            self.log_info("No need to convert, just Copy {} ---> {}".format(old_path, new_path))
            shutil.copyfile(old_path, new_path)

    def transfer_node(self, root, file):
        transformers = [
            ImportTransformer, # import ast transformer
            BasicTransformer,    # basic api ast transformer
        ]
        for transformer in transformers:
            trans = transformer(root, file, self.imports_map, self.logger)
            trans.transform()
            self.torch_api_count += trans.torch_api_count
            self.success_api_count += trans.success_api_count
                