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
import collections
import json
import os
import re

import astor

from paconvert.utils import PaddleAuxFile, log_debug

json_file = os.path.dirname(__file__) + "/api_mapping.json"
with open(json_file, "r") as file:
    API_MAPPING = json.load(file)

json_file = os.path.dirname(__file__) + "/attribute_mapping.json"
with open(json_file, "r") as file:
    ATTRIBUTE_MAPPING = json.load(file)

TORCH_PACKAGE_LIST = [
    "torch",
    "mmseg",
    "mmcv",
    "detectron",
    "timm",
    "mmdet",
    "mmdet3d",
    "torchvision",
    "kornia",
    "fasttext",
    "pytorch_lightning",
    "jieba",
    "sentencepiece",
    "NLTK",
    "scikit-learn",
]


class BaseTransformer(ast.NodeTransformer):
    def __init__(self, root, file, imports_map, logger):
        self.root = root
        self.file = file
        self.file_name = os.path.basename(file)
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

    @property
    def parent_node(self):
        return self.node_stack[-2]

    def scope_body_index(self, level=-1):
        scope_node = self.scope_stack[level]

        # reverse find scope_node in node_stack
        lower = -1 * (len(self.node_stack) + 1)
        for i in range(-1, lower, -1):
            if self.node_stack[i] == scope_node:
                for index, node in enumerate(scope_node.body):
                    if node == self.node_stack[i + 1]:
                        return scope_node, "body", index

                if getattr(scope_node, "orelse", None):
                    for index, node in enumerate(scope_node.orelse):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "orelse", index

                if getattr(scope_node, "decorator_list", None):
                    for index, node in enumerate(scope_node.decorator_list):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "decorator_list", index

        return self.scope_body_index(-2)

    def record_scope(self, scope_body_index, node_list):
        if not isinstance(node_list, list):
            node_list = [node_list]

        if len(node_list) == 0:
            return
        scope_node = scope_body_index[0]
        body = scope_body_index[1]
        index = scope_body_index[2]

        if body in self.scope_insert_lines[scope_node]:
            if index in self.scope_insert_lines[scope_node][body]:
                origin_node_list = self.scope_insert_lines[scope_node][body][index]
                for node in node_list.copy():
                    # remove dumplicate node
                    for ele in origin_node_list:
                        if ast.dump(node) == ast.dump(ele):
                            node_list.remove(node)
                            break

                origin_node_list.extend(node_list)
            else:
                self.scope_insert_lines[scope_node][body].update({index: node_list})
        else:
            self.scope_insert_lines[scope_node][body] = {index: node_list}

        if len(node_list) > 0:
            log_debug(self.logger, "insert extra {} lines".format(len(node_list)))

    def insert_scope(self):
        # if multiple line, insert into scope node only One time
        for scope_node in self.scope_insert_lines:
            for body in self.scope_insert_lines[scope_node]:
                insert_lines = self.scope_insert_lines[scope_node][body]
                insert_lines = sorted(
                    insert_lines.items(), key=lambda x: x[0], reverse=True
                )
                for index, lines in insert_lines:
                    for line in lines[::-1]:
                        getattr(scope_node, body).insert(index, line)

    def insert_multi_node(self, node_list):
        if len(node_list) == 0:
            return
        import_nodes = []
        other_nodes = []
        for node in node_list:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
            elif "sys.path" in astor.to_source(node):
                import_nodes.append(node)
            else:
                other_nodes.append(node)

        self.record_scope((self.root, "body", 0), import_nodes)
        self.record_scope(self.scope_body_index(), other_nodes)

    def get_full_attr(self, node):
        # torch.nn.fucntional.relu
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + "." + node.attr
        # x.abs() -> 'x'
        elif isinstance(node, ast.Name):
            # array(1.) ...
            node_str = astor.to_source(node).strip("\n")
            for item in self.black_list:
                if item == node_str:
                    return "NonTorchClass"
            return node.id
        # 1. torch.abs(x).transpose(1, 0) -> 'torchClass'
        # 2. (x == y).transpose(1, 0) -> 'torchClass'
        # 3. (x + y).transpose(1, 0) -> 'torchClass'
        # 4. x[0].transpose(1, 0) -> 'torchClass'
        # 5. (-x).transpose(1, 0) -> 'torchClass'
        elif isinstance(
            node, (ast.Call, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Subscript)
        ):
            node_str = astor.to_source(node).strip("\n")
            for item in self.black_list:
                # (array(1.) + array(2.)).abs() ...
                if re.match(".*[^A-Za-z_]{1}%s\(" % item, node_str):
                    return "NonTorchClass"
                # np.array(1.).abs() ...
                if re.match("%s\." % item, node_str):
                    return "NonTorchClass"
                # array(1.).abs() ...
                if re.match("%s\(" % item, node_str):
                    return "NonTorchClass"

            return "TorchClass"
        # others not torch, such as 'str'.split
        else:
            return "NonTorchClass"

    def get_full_api_from_node(self, node):
        full_attr = self.get_full_attr(node)
        attr_list = full_attr.split(".")
        old_module = attr_list[0]
        if old_module in self.imports_map[self.file]:
            new_module = self.imports_map[self.file][old_module]
            attr_list[0] = new_module
            torch_api = ".".join(attr_list)
            return torch_api
        else:
            return None


class BaseMatcher(object):
    def __init__(self, torch_api, api_mapping, logger):
        self.torch_api = torch_api
        self.paddle_api = None
        self.api_mapping = api_mapping
        self.logger = logger

    def get_aux_dir(self):
        return os.path.dirname(PaddleAuxFile().fileName)

    def get_paddle_api(self):
        if self.paddle_api:
            return self.paddle_api
        if "paddle_api" in self.api_mapping:
            return self.api_mapping["paddle_api"]
        return None

    def set_paddle_api(self, paddle_api):
        self.paddle_api = paddle_api

    def parse_args_and_kwargs(self, args, kwargs):
        args_list = self.api_mapping.get("args_list") or []
        # more args, not match torch class method, indicate it is not torch Class
        if len(args) > len(args_list):
            return "NonTorchClass"

        unsupport_args = self.api_mapping.get("unsupport_args") or []

        new_kwargs = {}
        for i, node in enumerate(args):
            # not support 'torch.rot90(tensor, *config)'
            if isinstance(node, ast.Starred):
                return None
            k = args_list[i]
            # not support some API args
            if k in unsupport_args:
                return None
            v = astor.to_source(node).strip("\n")
            # have comma indicates a tuple
            new_kwargs[k] = v

        for node in kwargs:
            k = node.arg
            # not support 'torch.rot90(tensor, **config)'
            if k is None:
                return None
            # not support some API args
            if k in unsupport_args:
                return None
            # TODO: will open after all args have been add in args_list
            # if k not in args_list:
            #    return 'NonTorchClass'
            v = astor.to_source(node.value).strip("\n")
            new_kwargs[k] = v

        new_kwargs = self.set_paddle_default_kwargs(new_kwargs)
        return new_kwargs

    def parse_args(self, args):
        new_args = []
        for node in args:
            ele = astor.to_source(node).strip("\n")
            new_args.append(ele)

        return new_args

    def parse_kwargs(self, kwargs):
        unsupport_args = self.api_mapping.get("unsupport_args") or []

        new_kwargs = {}
        for node in kwargs:
            k = node.arg
            if k in unsupport_args:
                return None
            v = astor.to_source(node.value).strip("\n")
            new_kwargs[k] = v

        new_kwargs = self.set_paddle_default_kwargs(new_kwargs)
        return new_kwargs

    def parse_func(self, func):
        new_func = astor.to_source(func).strip("\n")
        self.paddleClass = new_func[0 : new_func.rfind(".")]
        if self.get_paddle_api():
            new_paddle_api = re.sub(
                "paddle.Tensor|paddle.nn.Layer|paddle.optimizer.Optimizer",
                self.paddleClass,
                self.get_paddle_api(),
            )
            self.set_paddle_api(new_paddle_api)

        return new_func

    def args_to_str(self, args):
        str_list = []
        for ele in args:
            str_list.append("{}".format(ele))

        return ", ".join(str_list)

    def kwargs_to_str(self, kwargs):
        str_list = []
        for k, v in kwargs.items():
            str_list.append("{}={}".format(k, v))

        return ", ".join(str_list)

    def args_and_kwargs_to_str(self, args, kwargs):
        str_list = []
        for ele in args:
            str_list.append("{}".format(ele))

        for k, v in kwargs.items():
            str_list.append("{}={}".format(k, v))

        return ", ".join(str_list)

    def get_full_attr(self, node):
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + "." + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "None"

    def set_paddle_default_kwargs(self, kwargs):
        """
        process the redundant parameters of Paddle and set the default values
        and return the new parameter list in the form of a dictionary.
        """
        if "paddle_default_kwargs" in self.api_mapping:
            paddle_default_kwargs = self.api_mapping["paddle_default_kwargs"]
            for k in paddle_default_kwargs:
                if k not in kwargs:
                    kwargs[k] = paddle_default_kwargs[k]

        return kwargs

    def generate_aux_code(self):
        return None

    def write_aux_code(self):
        aux_code = self.generate_aux_code()
        if aux_code:
            log_debug(self.logger, "Write auxiliary code for {}".format(self.torch_api))
            PaddleAuxFile().write_code(aux_code, self.torch_api)

    @staticmethod
    def generate_code(self, kwargs):
        return None

    def get_paddle_nodes(self, args, kwargs):
        new_kwargs = self.parse_args_and_kwargs(args, kwargs)
        if new_kwargs is not None:
            new_code = self.generate_code(new_kwargs)
            if new_code:
                return ast.parse(new_code).body
        return None

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        new_kwargs = self.parse_args_and_kwargs(args, kwargs)
        # NonTorchClass means This API usage not match torch.Tensor/Module/Optimizer, so it is not a torch Class
        if new_kwargs == "NonTorchClass":
            return "NonTorchClass"
        elif new_kwargs is not None:
            new_code = self.generate_code(new_kwargs)
            if new_code == "NonTorchClass":
                return "NonTorchClass"
            elif new_code:
                return ast.parse(new_code).body

        return None

    def get_paddle_class_attribute_nodes(self, node):
        return None
