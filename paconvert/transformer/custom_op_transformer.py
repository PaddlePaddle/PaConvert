# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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

from paconvert.base import BaseTransformer
from paconvert.utils import log_debug, log_info

CPP_EXTENSION_LIST = []
AUTOGRAD_FUNC_NODES = {}


class PreCustomOpTransformer(BaseTransformer):
    def __init__(
        self, root, file, imports_map, logger, all_api_map=None, unsupport_api_map=None
    ):
        super(PreCustomOpTransformer, self).__init__(
            root, file, imports_map, logger, all_api_map, unsupport_api_map
        )
        self.cpp_ext_import_names = {}
        self.cpp_ext_load_names = []

    def visit_Import(self, node):
        new_node_names = []
        for alias_node in node.names:
            if alias_node.name in CPP_EXTENSION_LIST:
                if alias_node.asname:
                    log_info(
                        self.logger,
                        "remove 'import {} as {}' ".format(
                            alias_node.name, alias_node.asname
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    self.cpp_ext_import_names[alias_node.asname] = alias_node.name
                else:
                    log_info(
                        self.logger,
                        "remove 'import {}' ".format(alias_node.name),
                        self.file_name,
                        node.lineno,
                    )
                    self.cpp_ext_import_names[alias_node.name] = alias_node.name
            else:
                new_node_names.append(alias_node)

        if len(new_node_names) > 0:
            node.names = new_node_names
            return node
        else:
            return None

    def visit_Assign(self, node):
        super(PreCustomOpTransformer, self).generic_visit(node)
        if isinstance(node.value, ast.Call):
            if "paddle.utils.cpp_extension.load" == self.get_full_attr(node.value.func):
                self.cpp_ext_load_names.append(node.targets[0].id)
        return node

    def visit_Attribute(self, node):
        for cpp_ext_asname, cpp_ext in self.cpp_ext_import_names.items():
            if isinstance(node.value, ast.Name) and cpp_ext_asname == node.value.id:
                lower = -1 * (len(self.node_stack) + 1)
                for i in range(-1, lower, -1):
                    if isinstance(self.node_stack[i], ast.ClassDef):
                        for base in self.node_stack[i].bases:
                            if "paddle.autograd.PyLayer" == self.get_full_attr(base):
                                self.torch_api_count += 1
                                self.success_api_count += 1
                                log_debug(
                                    self.logger,
                                    "Start convert usage of cpp extension in torch.autograd.Function: {} to Paddle --> ".format(
                                        self.get_full_attr(node)
                                    ),
                                    self.file_name,
                                    node.lineno,
                                )
                                log_debug(
                                    self.logger,
                                    "[Delete] find usage of cpp extension in torch.autograd.Function: {}".format(
                                        self.get_full_attr(node)
                                    ),
                                    self.file_name,
                                    node.lineno,
                                )
                                AUTOGRAD_FUNC_NODES[self.node_stack[i]] = cpp_ext

        for cpp_ext_load in self.cpp_ext_load_names:
            if isinstance(node.value, ast.Name) and cpp_ext_load == node.value.id:
                lower = -1 * (len(self.node_stack) + 1)
                for i in range(-1, lower, -1):
                    if isinstance(self.node_stack[i], ast.ClassDef):
                        for base in self.node_stack[i].bases:
                            if "paddle.autograd.PyLayer" == self.get_full_attr(base):
                                self.torch_api_count += 1
                                self.success_api_count += 1
                                log_debug(
                                    self.logger,
                                    "Start convert usage of cpp extension in torch.autograd.Function: {} to Paddle --> ".format(
                                        self.get_full_attr(node)
                                    ),
                                    self.file_name,
                                    node.lineno,
                                )
                                log_debug(
                                    self.logger,
                                    "[Delete] find usage of cpp extension in torch.autograd.Function: {}".format(
                                        self.get_full_attr(node)
                                    ),
                                    self.file_name,
                                    node.lineno,
                                )
                                AUTOGRAD_FUNC_NODES[self.node_stack[i]] = (
                                    "load_" + cpp_ext_load
                                )

        return node


class CustomOpTransformer(BaseTransformer):
    def __init__(
        self, root, file, imports_map, logger, all_api_map=None, unsupport_api_map=None
    ):
        super(CustomOpTransformer, self).__init__(
            root, file, imports_map, logger, all_api_map, unsupport_api_map
        )
        self.autograd_func_import_names = {}

    def visit_ImportFrom(self, node):
        new_node_names = []
        import_list = []
        for alias_node in node.names:
            remove = False
            for autograd_func_node, cpp_ext in AUTOGRAD_FUNC_NODES.items():
                if autograd_func_node.name == alias_node.name:
                    if alias_node.asname:
                        self.autograd_func_import_names[alias_node.asname] = cpp_ext
                    else:
                        self.autograd_func_import_names[alias_node.name] = cpp_ext

                    if cpp_ext.startswith("load_"):
                        log_info(
                            self.logger,
                            f"change 'from {node.module} import {alias_node.name}' to 'from {node.module} import {cpp_ext.lstrip('load_')}' ",
                            self.file_name,
                            node.lineno,
                        )
                        alias_node.name = cpp_ext.lstrip("load_")
                    else:
                        log_info(
                            self.logger,
                            f"change 'from {node.module} import {alias_node.name}' to 'import {cpp_ext}' ",
                            self.file_name,
                            node.lineno,
                        )
                        import_list.append(cpp_ext)
                        remove = True

            if not remove:
                new_node_names.append(alias_node)

        if len(import_list) > 0:
            import_node = ast.parse("import " + ", ".join(import_list))
            self.record_scope((self.root, "body", 0), import_node.body)

        if len(new_node_names) > 0:
            node.names = new_node_names
            return node
        else:
            return None

    def visit_ClassDef(self, node):
        super(CustomOpTransformer, self).generic_visit(node)
        for autograd_func_node, cpp_ext in AUTOGRAD_FUNC_NODES.items():
            if autograd_func_node == node:
                log_debug(
                    self.logger,
                    "Just remove Class Definition {} ".format(autograd_func_node.name),
                    self.file_name,
                    node.lineno,
                )
                return None
        return node

    def visit_Attribute(self, node):
        for autograd_func_asname, cpp_ext in self.autograd_func_import_names.items():
            if f"{autograd_func_asname}.apply" == self.get_full_attr(node):
                self.torch_api_count += 1
                log_debug(
                    self.logger,
                    "Start convert {} to Paddle --> ".format(
                        f"{autograd_func_asname}.apply"
                    ),
                    self.file_name,
                    node.lineno,
                )
                if cpp_ext.startswith("load_"):
                    new_node = ast.parse(f"{cpp_ext.lstrip('load_')}").body[0]
                else:
                    new_node = ast.parse(f"{cpp_ext}.custom_op_xxx").body[0]

                self.unsupport_api_map[f"{autograd_func_asname}.apply"] += 1
                log_debug(
                    self.logger,
                    f"[Not Support] C++ Custom OP '{autograd_func_asname}.apply', only convert Python part and C++ part not support to convert",
                    self.file_name,
                    node.lineno,
                )
                annotate_node = ast.parse(
                    "'C++ Custom OP: only convert Python Code, Not Support auto convert C++ Code, please convert it by yourself'"
                ).body[0]
                self.record_scope(self.scope_body_index(), annotate_node)
                return new_node

        return node
