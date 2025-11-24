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
import os
import re
import textwrap
from itertools import groupby

import astor

from paconvert.utils import UtilsFileHelper, log_debug


class BaseTransformer(ast.NodeTransformer):
    def __init__(
        self, root, file, imports_map, logger, all_api_map=None, unsupport_api_map=None
    ):
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
        self.all_api_map = all_api_map
        self.unsupport_api_map = unsupport_api_map

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

                # else in (if scope)
                if getattr(scope_node, "orelse", None):
                    for index, node in enumerate(scope_node.orelse):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "orelse", index

                # decorator in (function scope)
                if getattr(scope_node, "decorator_list", None):
                    for index, node in enumerate(scope_node.decorator_list):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "decorator_list", index

                # finnally in (try scope)
                if getattr(scope_node, "finalbody", None):
                    for index, node in enumerate(scope_node.finalbody):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "finalbody", index

        return self.scope_body_index(-2)

    def insert_scope(self):
        # if multiple line, insert into scope node only One time
        for scope_node in self.scope_insert_lines:
            for body in self.scope_insert_lines[scope_node]:
                insert_lines = self.scope_insert_lines[scope_node][body]
                insert_lines = sorted(
                    insert_lines.items(), key=lambda x: x[0], reverse=True
                )
                for index, lines in insert_lines:
                    log_debug(
                        self.logger,
                        "insert extra {} lines finally".format(len(lines)),
                        self.file_name,
                    )
                    for line in lines[::-1]:
                        try:
                            getattr(scope_node, body).insert(index, line)
                        except:
                            getattr(scope_node, body).insert(0, line)

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
                    # remove duplicate node
                    for ele in origin_node_list:
                        if ast.dump(node) == ast.dump(ele):
                            node_list.remove(node)
                            break

                origin_node_list.extend(node_list)
            else:
                self.scope_insert_lines[scope_node][body].update({index: node_list})
        else:
            self.scope_insert_lines[scope_node][body] = {index: node_list}

    def insert_multi_node(self, node_list):
        if len(node_list) == 0:
            return True

        import_nodes = []
        other_nodes = []
        for node in node_list:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
            elif "sys.path" in astor.to_source(node):
                import_nodes.append(node)
            else:
                other_nodes.append(node)

        if len(import_nodes) > 0:
            self.record_scope((self.root, "body", 0), import_nodes)

        if len(other_nodes) > 0:
            if isinstance(self.parent_node, (ast.DictComp, ast.ListComp)):
                return False
            self.record_scope(self.scope_body_index(), other_nodes)

        return True

    def get_full_attr(self, node):
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + "." + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "None"

    def get_full_attr_for_apiname(self, node):
        if len(self.imports_map[self.file]["torch_packages"]) == 0:
            return "NonTorchClass"
        # x.abs() -> 'abs'
        if isinstance(node, ast.Attribute):
            for item in self.black_list:
                if item == node.attr:
                    return "NonTorchClass"
            return self.get_full_attr_for_apiname(node.value) + "." + node.attr
        # x.abs() -> 'x'
        elif isinstance(node, ast.Name):
            for item in self.black_list:
                if item == node.id:
                    return "NonTorchClass"
            # avoid to convert paddle itself, eg: paddle.max
            if "paddle" == node.id:
                return "NonTorchClass"
            return node.id
        # 1. torch.abs(x).transpose(1, 0) -> 'TorchClass'
        # 2. (x == y).transpose(1, 0) -> 'TorchClass'
        # 3. (x + y).transpose(1, 0) -> 'TorchClass'
        # 4. x[0].transpose(1, 0) -> 'TorchClass'
        # 5. (-x).transpose(1, 0) -> 'TorchClass'
        elif isinstance(
            node,
            (ast.Call, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Assert),
        ):
            node_str = astor.to_source(node).replace("\n", "")
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
        full_attr = self.get_full_attr_for_apiname(node)
        attr_list = full_attr.split(".")
        old_module = attr_list[0]
        if old_module in self.imports_map[self.file]:
            new_module = self.imports_map[self.file][old_module]
            attr_list[0] = new_module
            torch_api = ".".join(attr_list)
            return torch_api
        else:
            return None

    def visit_FunctionDef(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_While(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_If(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Try(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_TryFinally(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_For(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_With(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_ExceptHandler(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Module(self, node):
        self.scope_stack.append(node)
        super(BaseTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node


class BaseMatcher(object):
    def __init__(self, transformer, torch_api, api_mapping_dict, logger):
        self.transformer = transformer
        self.torch_api = torch_api
        self.paddle_api = None
        self.api_mapping_dict = api_mapping_dict
        self.logger = logger

    def parse_args_and_kwargs(
        self, args, kwargs, allow_starred=False, allow_none=False
    ):
        args_list = self.api_mapping_dict.get("args_list") or []
        # torch.dsplit has overload args
        overload_args_list = self.api_mapping_dict.get("overload_args_list") or []
        min_input_args_num = self.api_mapping_dict.get("min_input_args") or 0
        unsupport_args = self.api_mapping_dict.get("unsupport_args") or []

        group_list = [
            list(v) for k, v in groupby(args_list, lambda x: x == "*") if not k
        ]
        posion_args_list = group_list[0] if len(group_list) > 0 else []
        force_kwargs_list = group_list[1] if len(group_list) > 1 else []
        force_kwargs_num = 0
        for node in args:
            # not support 'torch.rot90(tensor, *config)'
            if isinstance(node, ast.Starred) and not allow_starred:
                return None
        for node in kwargs:
            k = node.arg
            # not support 'torch.rot90(tensor, **config)'
            if k is None and not allow_none:
                return None
            # not support some API args
            if k in unsupport_args:
                if isinstance(node.value, ast.Constant):
                    if node.value.value is None:
                        kwargs.remove(node)
                        continue
                return None
            if k not in args_list + overload_args_list:
                return "misidentify"
            if k in force_kwargs_list:
                force_kwargs_num += 1

        posion_args_num = len(args) + len(kwargs) - force_kwargs_num
        if posion_args_num < min_input_args_num:
            return "misidentify"
        if posion_args_num > len(posion_args_list):
            return "misidentify"

        new_kwargs = {}
        for i, node in enumerate(args):
            k = posion_args_list[i]
            # not support some API args
            if k in unsupport_args:
                return None
            v = astor.to_source(node).replace("\n", "")
            # v = ast.unparse(node)
            new_kwargs[k] = v

        for node in kwargs:
            k = node.arg
            if k in new_kwargs:
                log_debug(
                    self.logger,
                    f"Parameter '{k}' specified multiple times - cannot be both positional and keyword argument",
                    self.transformer.file_name,
                )
            v = astor.to_source(node.value).replace("\n", "")
            # v = ast.unparse(node.value)
            new_kwargs[k] = v

        return new_kwargs

    def parse_args(self, args):
        new_args = []
        for node in args:
            # if isinstance(node, ast.Starred) and not allow_starred:
            #    return None
            ele = astor.to_source(node).replace("\n", "")
            new_args.append(ele)

        return new_args

    def parse_kwargs(self, kwargs, allow_none=False):
        unsupport_args = self.api_mapping_dict.get("unsupport_args") or []

        new_kwargs = {}
        for node in kwargs:
            k = node.arg
            # not support 'torch.rot90(tensor, **config)'
            if k is None and not allow_none:
                return None
            # not support some API args
            if k in unsupport_args:
                return None
            v = astor.to_source(node.value).replace("\n", "")
            new_kwargs[k] = v

        return new_kwargs

    def parse_func(self, func):
        new_func = astor.to_source(func).replace("\n", "")
        self.paddleClass = new_func[0 : new_func.rfind(".")]
        if self.get_paddle_api():
            new_paddle_api = re.sub(
                "paddle.Tensor|paddle.nn.Module|paddle.optimizer.Optimizer|paddle.distribution.Distribution|paddle.autograd.PyLayerContext|paddle.profiler.Profiler",
                re.escape(self.paddleClass),
                self.get_paddle_api(),
            )
            # reverse escape
            new_paddle_api = re.sub(r"\\(.)", r"\1", new_paddle_api)
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
            if k is None:
                # 'torch.rot90(tensor, **config)'
                str_list.append("**{}".format(v))
            else:
                str_list.append("{}={}".format(k, v))

        return ", ".join(str_list)

    def change_kwargs(self, kwargs, unuse_args=[]):
        new_kwargs = {}
        kwargs_change = self.api_mapping_dict.get("kwargs_change", {})
        for k in kwargs.keys():
            if k in kwargs_change:
                if kwargs_change[k]:
                    if isinstance(kwargs_change[k], list):
                        for v in kwargs_change[k]:
                            new_kwargs[v] = kwargs[k]
                    else:
                        new_kwargs[kwargs_change[k]] = kwargs[k]
            else:
                new_kwargs[k] = kwargs[k]
                if k in unuse_args:
                    new_kwargs.pop(k)
        return new_kwargs

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
        if "paddle_default_kwargs" in self.api_mapping_dict:
            paddle_default_kwargs = self.api_mapping_dict["paddle_default_kwargs"]
            for k in paddle_default_kwargs:
                if k not in kwargs:
                    kwargs[k] = paddle_default_kwargs[k]

        return kwargs

    def generate_utils_code(self):
        return None

    def enable_utils_code(self):
        utils_code = self.generate_utils_code()
        if utils_code:
            utils_file_helper = UtilsFileHelper()
            log_debug(
                self.logger,
                "When convert {}, write utils code to file: {}".format(
                    self.torch_api, utils_file_helper.fileName
                ),
            )
            if utils_file_helper.is_dir_mode:
                CODE_TEMPLATE = textwrap.dedent(
                    """
                    import sys
                    sys.path.append(r'{}')
                    from paddle_utils import * # noqa: F403
                    """
                )
                # TODO: change sys.path.append to relative import
                code = CODE_TEMPLATE.format(os.path.dirname(utils_file_helper.fileName))
                self.transformer.insert_multi_node(ast.parse(code).body)

            utils_file_helper.add_code(utils_code)
            log_debug(self.logger, "add 'import utils'", self.transformer.file_name)

    def set_paddle_api(self, paddle_api):
        self.paddle_api = paddle_api

    def get_paddle_api(self):
        paddle_api = None
        if self.paddle_api:
            paddle_api = self.paddle_api
        elif "paddle_api" in self.api_mapping_dict:
            paddle_api = self.api_mapping_dict["paddle_api"]
        if (
            paddle_api
            and self.api_mapping_dict.get("abstract")
            and self.generate_utils_code() is not None
        ):
            self.enable_utils_code()
        if self.api_mapping_dict.get("enable_utils_code"):
            self.enable_utils_code()
        return paddle_api

    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "{}".format(self.paddle_api)
        return ast.parse(code).body

    @staticmethod
    def generate_code(self, kwargs):
        if self.api_mapping_dict.get("enable_utils_code"):
            self.enable_utils_code()
        return None

    def get_paddle_nodes(self, args, kwargs):
        new_kwargs = self.parse_args_and_kwargs(args, kwargs)
        if new_kwargs == "misidentify":
            return "misidentify"
        elif new_kwargs is not None:
            new_code = self.generate_code(new_kwargs)
            if new_code == "misidentify":
                return "misidentify"
            elif new_code == "unchange":
                return "unchange"
            elif new_code:
                return ast.parse(new_code).body
        return None

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        return self.get_paddle_nodes(args, kwargs)
