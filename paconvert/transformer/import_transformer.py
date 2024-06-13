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
import os

from paconvert.base import ALIAS_MAPPING, BaseTransformer
from paconvert.utils import log_info

from ..base import MAY_TORCH_PACKAGE_LIST, TORCH_PACKAGE_LIST


class ImportTransformer(BaseTransformer):
    """
    Record import information
    """

    def __init__(self, root, file, imports_map, logger, unsupport_map=None):
        super(ImportTransformer, self).__init__(
            root, file, imports_map, logger, unsupport_map
        )
        self.imports_map[self.file]["other_packages"] = []
        self.imports_map[self.file]["torch_packages"] = []
        self.imports_map[self.file]["alias_call_map"] = {}
        self.import_paddle = False
        self.import_paddlenlp = False
        self.import_MAY_TORCH_PACKAGE_LIST = []
        self.ast_if_List = []

    def visit_Import(self, node):
        """
        1. remove import torch.nn
        2. remove import torch.nn as nn
        3. record whether to import paddle
        """
        new_node_names = []
        for alias_node in node.names:
            has_done = False

            # import from current project
            dir_name = os.path.dirname(self.file)
            """
            while (
                len(dir_name) > 1 and dir_name[-2] != ":"
            ):  # the case of dir_name = 'E:/' will happen with windows
                import_path = os.path.join(dir_name, alias_node.name.replace(".", "/"))

                if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                    self.insert_other_packages(self.imports_map, alias_node)
                    new_node_names.append(alias_node)
                    has_done = True
                    break

                dir_name = os.path.dirname(dir_name)
            """
            import_path = os.path.join(dir_name, alias_node.name.replace(".", "/"))
            if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                self.insert_other_packages(self.imports_map, alias_node)
                new_node_names.append(alias_node)
                has_done = True
                break

            if has_done:
                continue

            # import from torch
            for pkg_name in TORCH_PACKAGE_LIST + MAY_TORCH_PACKAGE_LIST:
                if f"{pkg_name}." in alias_node.name or pkg_name == alias_node.name:
                    if pkg_name in MAY_TORCH_PACKAGE_LIST:
                        if pkg_name not in self.import_MAY_TORCH_PACKAGE_LIST:
                            self.import_MAY_TORCH_PACKAGE_LIST.append(pkg_name)
                    else:
                        self.imports_map[self.file]["torch_packages"].append(pkg_name)
                        self.import_paddle = True
                        if pkg_name == "transformers":
                            self.import_paddlenlp = True
                    if alias_node.asname:
                        if (
                            isinstance(self.parent_node, ast.If)
                            and self.parent_node not in self.ast_if_List
                        ):
                            self.insert_multi_node([ast.parse("pass")])
                            self.ast_if_List.append(self.parent_node)
                        log_info(
                            self.logger,
                            "remove 'import {} as {}' ".format(
                                alias_node.name, alias_node.asname
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        self.imports_map[self.file][alias_node.asname] = alias_node.name
                    else:
                        if (
                            isinstance(self.parent_node, ast.If)
                            and self.parent_node not in self.ast_if_List
                        ):
                            self.insert_multi_node([ast.parse("pass")])
                            self.ast_if_List.append(self.parent_node)
                        log_info(
                            self.logger,
                            "remove 'import {}' ".format(alias_node.name),
                            self.file_name,
                            node.lineno,
                        )
                        self.imports_map[self.file][alias_node.name] = alias_node.name
                    has_done = True
                    break

            if has_done:
                continue

            # other_packages
            self.insert_other_packages(self.imports_map, alias_node)
            new_node_names.append(alias_node)

        if len(new_node_names) > 0:
            node.names = new_node_names
            return node
        else:
            return None

    def visit_ImportFrom(self, node):
        """
        1. remove from torch import nn
        2. remove from torch import nn.functional as F
        """
        # import from current project
        if node.level > 0:
            # from ..datasets import xxx
            # from ... import xxx (node.module is None)
            """
            import_path = os.path.dirname(self.file) + "../" * (node.level-1)
            if node.module:
                import_path = os.path.join(import_path, node.module.replace(".", "/"))

            if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                return node
            """
            self.insert_other_packages(self.imports_map, node)
            return node
        else:
            # from yolov3.datasets import xxx
            # from datasets import xxx
            dir_name = os.path.dirname(self.file)
            """
            while (
                len(dir_name) > 1 and dir_name[-2] != ":"
            ):  # the case of dir_name = 'E:/' will happen with windows
                import_path = os.path.join(dir_name, node.module.replace(".", "/"))

                if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                    self.insert_other_packages(self.imports_map, node)
                    return node

                dir_name = os.path.dirname(dir_name)
            """
            import_path = os.path.join(dir_name, node.module.replace(".", "/"))
            if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                self.insert_other_packages(self.imports_map, node)
                return node

        # import from TORCH_PACKAGE_LIST
        for pkg_name in TORCH_PACKAGE_LIST + MAY_TORCH_PACKAGE_LIST:
            if f"{pkg_name}." in node.module or pkg_name == node.module:
                if pkg_name in TORCH_PACKAGE_LIST:
                    self.imports_map[self.file]["torch_packages"].append(pkg_name)
                    self.import_paddle = True
                    if pkg_name == "transformers":
                        self.import_paddlenlp = True
                else:
                    if pkg_name not in self.import_MAY_TORCH_PACKAGE_LIST:
                        self.import_MAY_TORCH_PACKAGE_LIST.append(pkg_name)
                for alias_node in node.names:
                    if alias_node.asname:
                        if (
                            isinstance(self.parent_node, ast.If)
                            and self.parent_node not in self.ast_if_List
                        ):
                            self.insert_multi_node([ast.parse("pass")])
                            self.ast_if_List.append(self.parent_node)
                        log_info(
                            self.logger,
                            "remove 'from {} import {} as {}' ".format(
                                node.module, alias_node.name, alias_node.asname
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        self.imports_map[self.file][alias_node.asname] = ".".join(
                            [node.module, alias_node.name]
                        )
                    else:
                        if (
                            isinstance(self.parent_node, ast.If)
                            and self.parent_node not in self.ast_if_List
                        ):
                            self.insert_multi_node([ast.parse("pass")])
                            self.ast_if_List.append(self.parent_node)
                        log_info(
                            self.logger,
                            "remove 'from {} import {}' ".format(
                                node.module, alias_node.name
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        self.imports_map[self.file][alias_node.name] = ".".join(
                            [node.module, alias_node.name]
                        )
                return None

        # other_packages
        self.insert_other_packages(self.imports_map, node)
        return node

    def insert_other_packages(self, imports_map, node):
        if isinstance(node, ast.ImportFrom):
            for alias_node in node.names:
                if alias_node.asname:
                    self.imports_map[self.file]["other_packages"].append(
                        alias_node.asname
                    )
                else:
                    # from data_loader.modules import *
                    if alias_node.name != "*":
                        self.imports_map[self.file]["other_packages"].append(
                            alias_node.name
                        )
        elif isinstance(node, ast.alias):
            if node.asname:
                self.imports_map[self.file]["other_packages"].append(node.asname)
            else:
                self.imports_map[self.file]["other_packages"].append(node.name)

    def visit_Attribute(self, node):
        """
        change torch api to full api according to import info.
        eg.
            nn.Module -> torch.nn.Module
        """
        if isinstance(
            node.value, (ast.Call, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Subscript)
        ):
            super(ImportTransformer, self).generic_visit(node)

        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            if torch_api in ALIAS_MAPPING:
                torch_api = ALIAS_MAPPING[torch_api]
            return ast.parse(torch_api).body[0].value
        return node

    def visit_Name(self, node):
        """
        change torch api name to full api according to import info.
        eg.
            from torch.nn import Module
            from torch import Tensor
            from torch import float32
            import torch.add as TorchAdd

            1. class A(Module):
            2. def func() -> Tensor:
            3. def func(x: Tensor):
            4. def func(dtype=float32):
            5. Tensor(2, 3)
            6. isinstance(x, Tensor)
            7. setattr(Tensor, 'add', func)
            8. {'build_ext': BuildExtension}
            9. inputs: Optional[Tensor] = None
            10. Union[GenerateOutput, torch.LongTensor]
            11. my_add = TorchAdd
            12. Union[List[str], List[AddedToken]],
            13. hasattr(torch, 'version')
        """
        is_torch = False
        is_alias_call = False
        if isinstance(
            self.parent_node,
            (
                ast.ClassDef,  # 1. ast.ClassDef(bases=[ast.Name])
                ast.FunctionDef,  # 2. ast.FunctionDef(returns=ast.Name)
                ast.arg,  # 3. ast.arg(args='x', annotation=ast.Name)
                ast.arguments,  # 4. ast.arguments(args=[ast.arg(args='dtype'], defaults=[ast.Name])
                ast.Dict,  # 5. ast.Dict(keys=[ast.Constant], values=[ast.Name])
            ),
        ):
            is_torch = True
        elif isinstance(self.parent_node, ast.Call) and isinstance(
            self.parent_node.func, ast.Name
        ):
            if self.parent_node.func == node:  # 5
                is_torch = True
            elif self.parent_node.func.id in [
                "isinstance",
                "setattr",
                "hasattr",
            ]:  # 6/7/13
                is_torch = True
        elif (
            isinstance(self.parent_node, ast.Subscript)
            and self.parent_node.slice == node
        ):
            is_torch = True  # 9. Optional[Tensor] = None
        elif (
            isinstance(self.parent_node, ast.Tuple)
            and len(self.node_stack) >= 3
            and isinstance(self.node_stack[-3], ast.Subscript)
        ):
            is_torch = True  # 10. Union[GenerateOutput, torch.LongTensor]
        elif (
            isinstance(self.parent_node, ast.Assign) and node == self.parent_node.value
        ):
            is_torch = True  # 11. my_add = TorchAdd
            # When the parent node is a ast.Assign, we need to check if the lvalue is an alias call
            is_alias_call = True
        elif isinstance(self.parent_node, ast.Index) and self.parent_node.value == node:
            is_torch = True  # 12. Union[List[str], List[AddedToken]]

        if is_torch:
            torch_api = self.get_full_api_from_node(node)
            if torch_api:
                if torch_api in ALIAS_MAPPING:
                    torch_api = ALIAS_MAPPING[torch_api]
                # When use is_alias_call, is_torch must be True
                if is_alias_call:
                    # node.targets is a list
                    if len(self.parent_node.targets) == 1:
                        self.imports_map[self.file]["alias_call_map"][
                            self.parent_node.targets[0].id
                        ] = torch_api
                return ast.parse(torch_api).body[0].value
        return node

    def visit_Module(self, node):
        """
        add import paddle
        """
        super(ImportTransformer, self).generic_visit(node)
        line_NO = 1
        if self.import_paddle:
            log_info(
                self.logger, f"add 'import paddle' in line {line_NO}", self.file_name
            )
            self.record_scope((self.root, "body", 0), ast.parse("import paddle").body)
            line_NO += 1
        if self.import_paddlenlp:
            log_info(
                self.logger, f"add 'import paddlenlp' in line {line_NO}", self.file_name
            )
            self.record_scope(
                (self.root, "body", 0), ast.parse("import paddlenlp").body
            )
            line_NO += 1
        if len(self.import_MAY_TORCH_PACKAGE_LIST) > 0:
            for package in self.import_MAY_TORCH_PACKAGE_LIST:
                log_info(
                    self.logger,
                    f"add 'import {package}' in line {line_NO}",
                    self.file_name,
                )
                self.record_scope(
                    (self.root, "body", 0), ast.parse(f"import {package}").body
                )
                line_NO += 1

    def visit_Call(self, node):
        # modify the alias call to the full api

        # Use Postorder traversal
        super(BaseTransformer, self).generic_visit(node)

        full_attr = self.get_full_attr(node.func)
        if full_attr in self.imports_map[self.file]["alias_call_map"]:
            node.func.id = self.imports_map[self.file]["alias_call_map"][node.func.id]
        return node
