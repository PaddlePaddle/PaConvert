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

import ast
import astor
import json
import collections
import re
from os import path

from paddleconverter.utils import UniqueNameGenerator

json_file = path.dirname(__file__) + "/api_mapping.json"
with open(json_file, 'r') as file:
    API_MAPPING = json.load(file)


class BaseTransformer(ast.NodeTransformer):
    def __init__(self, root, file, imports_map, logger):
        self.root = root
        self.file = file
        self.file_name = path.basename(file)
        self.imports_map = imports_map
        self.torch_api_count = 0
        self.success_api_count = 0
        self.root = root
        self.node_stack = []
        self.scope_stack = []
        self.scope_insert_lines = collections.defaultdict(dict)
        self.logger = logger
        self.black_list = []
    
    def transform(self):
        self.visit(self.root)
        self.insert_scope()

    def visit(self, node):
        self.node_stack.append(node)
        node = super(BaseTransformer, self).visit(node)
        self.node_stack.pop()
        return node

    def record_scope(self, scope_node_body_index, node):
        if node is None:
            return
        if not isinstance(node, list):
            node = [node]
        scope_node = scope_node_body_index[0]
        body = scope_node_body_index[1]
        index = scope_node_body_index[2]

        if body in self.scope_insert_lines[scope_node]:
            if index in self.scope_insert_lines[scope_node][body]:
                self.scope_insert_lines[scope_node][body][index].extend(node)
            else:
                self.scope_insert_lines[scope_node][body].update({index: node})
        else:
            self.scope_insert_lines[scope_node][body] = {index: node}
            
    def insert_scope(self):
        # if multiple line, insert into scope node only One time
        for scope_node in self.scope_insert_lines:
            for body in self.scope_insert_lines[scope_node]:
                insert_lines = self.scope_insert_lines[scope_node][body]
                insert_lines = sorted(insert_lines.items(), 
                                    key = lambda x:x[0], 
                                    reverse = True)
                for index, lines in insert_lines:
                    for line in lines[::-1]:
                        getattr(scope_node, body).insert(index, line)

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

    def get_full_attr(self, node):
        # torch.nn.fucntional.relu
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + '.' + node.attr
        # x.abs() -> 'x'
        elif isinstance(node, ast.Name):
            # array(1.) ... 
            node_str = astor.to_source(node)
            for item in self.black_list:
                if item == node_str:
                    return 'None'
            return node.id
        # 1. torch.abs(x).transpose(1, 0) -> 'torchTensor'
        # 2. (x == y).transpose(1, 0) -> 'torchTensor'
        # 3. (x + y).transpose(1, 0) -> 'torchTensor'
        # 4. x[0].transpose(1, 0) -> 'torchTensor'
        # 5. (-x).transpose -> 'torchTensor'
        elif isinstance(node, (ast.Call, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Subscript)):
            # np.array(1.).transpose(1, 0) ...
            # array(1.).transpose(1, 0) ...
            node_str = astor.to_source(node)
            for item in self.black_list:
                if re.match('[^A-Za-z]*' + item, node_str):
                    return 'None'
            
            return 'torchTensor'
        # others, such as 'str'.split
        else:
            return 'None'

    def get_full_api_from_node(self, node):
        full_attr = self.get_full_attr(node)
        attr_list = full_attr.split('.')
        old_module = attr_list[0]
        if old_module in self.imports_map[self.file]:
            new_module = self.imports_map[self.file][old_module]
            attr_list[0] = new_module
            torch_api = '.'.join(attr_list)
            return torch_api
        else:
            return None


class BaseMatcher(object):
    def __init__(self, torch_api, api_mapping):
        self.torch_api = torch_api
        self.paddle_api = None
        self.api_mapping = api_mapping

    def get_paddle_api(self):
        if self.paddle_api:
            return self.paddle_api
        if 'paddle_api' in self.api_mapping:
            return self.api_mapping['paddle_api']
        return None

    def set_paddle_api(self, paddle_api):
        self.paddle_api = paddle_api

    def parse_args_and_kwargs(self, args, kwargs):
        args_list = self.api_mapping.get('args_list') or []
        # assert len(args) <= len(args_list)
        # For: Tensor Method, this API usage is not match torch.Tensor, so it is not Tensor
        if len(args) > len(args_list):
            return 'NonTensor'

        new_kwargs = {}
        for i, node in enumerate(args):
            #TODO: try to support torch.rot90(tensor, *config)
            if isinstance(node, ast.Starred):
                return None
            k = args_list[i]
            v = astor.to_source(node).strip('\n')
            new_kwargs[k] = v
        
        for node in kwargs:
            k = node.arg
            #TODO: try to support torch.rot90(tensor, **config)
            if k is None:
                return None
            v = astor.to_source(node.value).strip('\n')
            new_kwargs[k] = v

        return new_kwargs

    def parse_args(self, args):
        new_args = []
        for node in args:
            ele = astor.to_source(node).strip('\n')
            new_args.append(ele)

        return new_args

    def parse_kwargs(self, kwargs):
        new_kwargs = {}
        for node in kwargs:
            k = node.arg
            v = astor.to_source(node.value).strip('\n')
            new_kwargs[k] = v

        return new_kwargs

    def parse_func(self, func):
        new_func = astor.to_source(func).strip('\n')
        self.paddleTensor = new_func[0: new_func.rfind('.')]
        if self.get_paddle_api():
            new_paddle_api = self.get_paddle_api().replace('paddle.Tensor', self.paddleTensor)
            self.set_paddle_api(new_paddle_api)
        return new_func

    def args_to_str(self, args):
        str_list = []
        for ele in args:
            str_list.append('{}'.format(ele))
        
        return ', '.join(str_list)
        
    def kwargs_to_str(self, kwargs):
        str_list = []
        for k, v in kwargs.items():
            str_list.append('{}={}'.format(k, v))

        return ', '.join(str_list)

    def args_and_kwargs_to_str(self, args, kwargs):
        str_list = []
        for ele in args:
            str_list.append('{}'.format(ele))
        
        for k, v in kwargs.items():
            str_list.append('{}={}'.format(k, v))

        return ', '.join(str_list)

    def get_full_attr(self, node):
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + '.' + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return 'None'

    def generate_code(self, kwargs):
        return None

    def get_paddle_nodes(self, args, kwargs):
        new_kwargs = self.parse_args_and_kwargs(args, kwargs)
        if new_kwargs is not None:
            new_code = self.generate_code(new_kwargs)
            if new_code:
                return ast.parse(new_code).body
        return None

    def get_paddle_tensor_nodes(self, func, args, kwargs):
        self.parse_func(func)
        new_kwargs = self.parse_args_and_kwargs(args, kwargs)
        # NonTensor means This API usage not match torch.Tensor, so it is not a Tensor
        if new_kwargs == "NonTensor":
            return "NonTensor"
        elif new_kwargs is not None:
            new_code = self.generate_code(new_kwargs)
            if new_code == "NonTensor":
                return "NonTensor"
            elif new_code is not None:
                return ast.parse(new_code).body
        
        return None
