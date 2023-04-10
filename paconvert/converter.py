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
import re


from .transformer.import_transformer import ImportTransformer
from .transformer.basic_transformer import BasicTransformer
from .transformer.tensor_requires_grad_transformer import TensorRequiresGradTransformer
from .base import TORCH_PACKAGE_LIST

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class Converter:
    def __init__(self, log_dir=None, log_level='INFO', show_unsupport=False):
        self.imports_map = collections.defaultdict(dict)
        self.torch_api_count = 0
        self.success_api_count = 0
        self.logger = logging.getLogger(name='Converter')
        if log_dir is None:
            self.logger.addHandler(logging.StreamHandler())
        elif log_dir == 'disable':
            logging.disable(log_level)
        else:
            self.logger.addHandler(logging.FileHandler(log_dir, mode='w'))
        self.logger.setLevel(log_level)
        self.show_unsupport = show_unsupport
        self.unsupport_map = collections.defaultdict(int)

        self.log_info("===========================================")
        self.log_info("PyTorch to Paddle Convert Start ------>:")
        self.log_info("===========================================")

    def run(self, in_dir, out_dir=None, exclude_dirs=None):
        in_dir = os.path.abspath(in_dir)
        if out_dir is None:
            out_dir = os.getcwd()+ '/paddle_project'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.abspath(out_dir)

        assert out_dir != in_dir, "--out_dir must be different from --in_dir"

        exclude_dir_list = []
        if exclude_dirs:
            exclude_dirs = exclude_dirs.split(',')
            for item in exclude_dirs:
                exclude_dir_list.append(os.path.abspath(item))

        self.transfer_dir(in_dir, out_dir, exclude_dir_list)
        
        if self.show_unsupport:
            unsupport_map = sorted(self.unsupport_map.items(),
                                    key = lambda x:x[1],
                                    reverse = True)
            self.log_info("\n========================================")
            self.log_info("Not Support API List:")
            self.log_info("========================================")
            self.log_info("These Pytorch APIs are not supported to convert to Paddle now, which will be supppored in future!\n")
            for k, v in unsupport_map:
                self.log_info("{}: {}".format(k, v))

        faild_api_count = self.torch_api_count - self.success_api_count
        self.log_info("\n========================================")
        self.log_info("Convert Summary:")
        self.log_info("========================================")
        self.log_info("There are {} Pytorch APIs in this Project:".format(self.torch_api_count))
        self.log_info(" {}  Pytorch APIs have been converted to Paddle successfully!".format(self.success_api_count))
        self.log_info(" {}  Pytorch APIs are not supported to convert to Paddle currently!".format(faild_api_count))
        if self.torch_api_count > 0:
            convert_rate = self.success_api_count/self.torch_api_count
        else:
            convert_rate = 0.
        self.log_info(" Convert Rate is: {:.3%}".format(convert_rate))
        if (faild_api_count > 0):
            self.log_info("\nFor these {} Pytorch APIs that do not support to Convert now, which have been marked by >>> before the line, \nplease refer to "
        "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html"
        " \nand convert it by yourself manually. In addition, these APIs will be supported in future.".format(faild_api_count))
        self.log_info("\nThank you to use Paddle Code Convert Tool. You can make any suggestions to us.\n")
        return self.success_api_count, faild_api_count

    def transfer_dir(self, in_dir, out_dir, exclude_dir_list):
        if os.path.isfile(in_dir):
            old_path = in_dir
            if exclude_dir_list:
                for exclude_dir in exclude_dir_list:
                    if old_path == exclude_dir or old_path.startswith(exclude_dir+'/'):
                        return

            if os.path.isdir(out_dir):
                new_path = os.path.join(out_dir, os.path.basename(old_path))
            else:
                new_path = out_dir
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            if not os.path.isdir(os.path.dirname(new_path)):
                os.remove(os.path.dirname(new_path))
                os.makedirs(os.path.dirname(new_path))
            self.transfer_file(old_path, new_path)
        elif os.path.isdir(in_dir):
            in_dir_item = listdir_nohidden(in_dir)
            for item in in_dir_item:
                old_path = os.path.join(in_dir, item)
                new_path = os.path.join(out_dir, item)
                
                is_exclude = False
                if exclude_dir_list:
                    for exclude_dir in exclude_dir_list:
                        if old_path == exclude_dir or old_path.startswith(exclude_dir+'/'):
                            is_exclude = True

                if is_exclude:
                    continue

                if os.path.isdir(old_path):
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                self.transfer_dir(old_path, new_path, exclude_dir_list)
        elif os.path.islink(in_dir):
            pass
            # may need to create link
        else:
            raise ValueError(" the input 'in_dir' must be a exist file or directory! ")

    def transfer_file(self, old_path, new_path):
        if old_path.endswith(".py"):
            self.log_info("Start convert {} --> {}".format(old_path, new_path))
            with open(old_path, 'r', encoding='UTF-8') as f:
                code = f.read()
                root = ast.parse(code)
            
            self.transfer_node(root, old_path)
            code = astor.to_source(root)
            code = self.mark_unsport(code)

            with open(new_path, 'w', encoding='UTF-8') as file:
                file.write(code)
            self.log_info("Finish convert {} --> {}\n".format(old_path, new_path))
        elif old_path.endswith("requirements.txt"):
            self.log_info("Start convert {} --> {}".format(old_path, new_path))
            with open(old_path, 'r', encoding='UTF-8') as old_file:
                code = old_file.read()
            code = code.replace('torch', 'paddlepaddle-gpu')
            with open(new_path, 'w', encoding='UTF-8') as new_file:
                new_file.write(code)
            self.log_info("Finish convert {} --> {}\n".format(old_path, new_path))
        else:
            self.log_info("No need to convert, just Copy {} --> {}\n".format(old_path, new_path))
            shutil.copyfile(old_path, new_path)

    def transfer_node(self, root, file):
        # import ast transformer
        import_trans = ImportTransformer(root, file, self.imports_map, self.logger)
        import_trans.transform()
        # basic api ast transformer
        if import_trans.import_paddle:
            # attribute requires_grad transformer
            attribute_requires_grad_trans = TensorRequiresGradTransformer(root, file, self.imports_map, self.logger)
            attribute_requires_grad_trans.transform()
            # api transformer
            api_trans = BasicTransformer(root, file, self.imports_map, self.logger, self.unsupport_map)
            api_trans.transform()
            self.torch_api_count += api_trans.torch_api_count
            self.success_api_count += api_trans.success_api_count
                
    def mark_unsport(self, code):
        lines = code.split('\n')
        mark_next_line = False
        in_str = False
        for i, line in enumerate(lines):
            # torch.* in __doc__
            # torch.* in str
            if line.count('\"\"\"') % 2 != 0:
                in_str = not in_str
            
            tmp_line = re.sub(r'[\'\"]{1}[^\'\"]+[\'\"]{1}', "", line)
            if in_str:
                continue

            if 'Class Method:' in line or 'Tensor Attribute:' in line:
                mark_next_line = True
                continue
            else:
                # func decorator_list: @
                if mark_next_line and line != '@':
                    lines[i] = ">>>" + line
                    mark_next_line = False
                    continue
            
            # model_torch.npy
            for torch_package in TORCH_PACKAGE_LIST:
                if tmp_line.startswith("%s." % torch_package):
                    lines[i] = ">>>" + line
                    break

                if re.match(r".*[^A-Za-z_]{1}%s\." % torch_package, tmp_line):
                    lines[i] = ">>>" + line
                    
        return '\n'.join(lines)

    def log_debug(self, msg, file=None, line=None):
        if file:
            if line:
                msg = "[{}:{}] {}".format(file, line, msg)
            else:
                msg = "[{}] {}".format(file, msg)
        else:
            msg = "{}".format(msg)
        self.logger.debug(msg)

    def log_info(self, msg, file=None, line=None):
        if file:
            if line:
                msg = "[{}:{}] {}".format(file, line, msg)
            else:
                msg = "[{}] {}".format(file, msg)
        else:
            msg = "{}".format(msg)
        self.logger.info(msg)
