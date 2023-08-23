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

    def __init__(self, root, file, imports_map, logger):
        super(ImportTransformer, self).__init__(root, file, imports_map, logger)
        self.imports_map[self.file]["other_packages"] = []
        self.import_paddle = False

    def visit_Import(self, node):
        """
        1. remove import torch.nn
        2. remove import torch.nn as nn
        3. record whether to import paddle
        """
        new_node_names = []
        for alias_node in node.names:
            belong_torch = False
            for torch_package in TORCH_PACKAGE_LIST:
                if (
                    "%s." % torch_package in alias_node.name
                    or torch_package == alias_node.name
                ):
                    belong_torch = True
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
                        self.imports_map[self.file][torch_package] = torch_package

            if not belong_torch:
                if alias_node.asname:
                    self.imports_map[self.file]["other_packages"].append(
                        alias_node.asname
                    )
                else:
                    self.imports_map[self.file]["other_packages"].append(
                        alias_node.name
                    )
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
        # from . import Net (node.module is None)
        if node.module:
            for torch_package in TORCH_PACKAGE_LIST:
                if "%s." % torch_package in node.module or torch_package == node.module:
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

        # import from this directory
        if node.level > 0:
            import_path = os.path.dirname(self.file)
            i = 1
            while i < node.level:
                import_path += "../"
                i += 1

            # from . import Net (node.module is None)
            if node.module:
                import_path = import_path + "/" + node.module.replace(".", "/")
            if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                return node
        else:
            dir_name = os.path.dirname(self.file)
            # the case of dir_name = 'E:/' will happen with windows
            while len(dir_name) > 1 and dir_name[-2] != ":":
                import_path = dir_name + "/" + node.module.replace(".", "/")
                if os.path.exists(import_path) or os.path.exists(import_path + ".py"):
                    return node
                dir_name = os.path.dirname(dir_name)

        # import from site-packages, third_party module, just add to others
        for alias_node in node.names:
            if alias_node.asname:
                self.imports_map[self.file]["other_packages"].append(alias_node.asname)
            else:
                # from data_loader.modules import *
                if alias_node.name != "*":
                    self.imports_map[self.file]["other_packages"].append(
                        alias_node.name
                    )

        return node

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
        change torch api to full api according to import info.
        eg.
            Module -> torch.nn.Module
        """
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
