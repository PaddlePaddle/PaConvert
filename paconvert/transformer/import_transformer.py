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

from paconvert.base import ALIAS_MAPPING, TORCH_PACKAGE_LIST, BaseTransformer
from paconvert.utils import log_info


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
        self.import_paddle = False
        self.import_setuptools = False

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
            if has_done:
                continue

            # import from torch
            for pkg_name in TORCH_PACKAGE_LIST + ["setuptools"]:
                if f"{pkg_name}." in alias_node.name or pkg_name == alias_node.name:
                    self.imports_map[self.file]["torch_packages"].append(pkg_name)
                    if pkg_name == "setuptools":
                        self.import_setuptools = True
                    else:
                        self.import_paddle = True
                    if alias_node.asname:
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
            while (
                len(dir_name) > 1 and dir_name[-2] != ":"
            ):  # the case of dir_name = 'E:/' will happen with windows
                import_path = os.path.join(dir_name, node.module.replace(".", "/"))

                if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                    self.insert_other_packages(self.imports_map, node)
                    return node

                dir_name = os.path.dirname(dir_name)

        # import from torch
        for pkg_name in TORCH_PACKAGE_LIST + ["setuptools"]:
            if f"{pkg_name}." in node.module or pkg_name == node.module:
                self.imports_map[self.file]["torch_packages"].append(pkg_name)
                if pkg_name == "setuptools":
                    self.import_setuptools = True
                else:
                    self.import_paddle = True
                for alias_node in node.names:
                    if alias_node.asname:
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

            1. class A(Module):
            2. def func() -> Tensor:
            3. def func(x: Tensor):
            4. def func(dtype=float32):
            5. Tensor(2, 3)
            6. isinstance(x, Tensor)
            7. setattr(Tensor, 'add', func)
        """
        is_torch = False
        if isinstance(
            self.parent_node,
            (ast.Call, ast.ClassDef, ast.FunctionDef, ast.arg, ast.arguments),
        ):
            is_torch = True
        elif isinstance(self.parent_node, ast.Call) and isinstance(
            self.parent_node.func, ast.Name
        ):
            if self.parent_node.func.id in ["isinstance", "setattr"]:
                is_torch = True

        if is_torch:
            torch_api = self.get_full_api_from_node(node)
            if torch_api:
                if torch_api in ALIAS_MAPPING:
                    torch_api = ALIAS_MAPPING[torch_api]
                return ast.parse(torch_api).body[0].value
        return node

    def visit_Module(self, node):
        """
        add import paddle
        """
        super(ImportTransformer, self).generic_visit(node)

        if self.import_paddle:
            log_info(self.logger, "add 'import paddle' in first line", self.file_name)
            self.record_scope((self.root, "body", 0), ast.parse("import paddle").body)

        if self.import_setuptools:
            log_info(
                self.logger, "add 'import setuptools' in second line", self.file_name
            )
            self.record_scope(
                (self.root, "body", 1), ast.parse("import setuptools").body
            )
