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

import ast
import os

from paconvert.base import BaseTransformer
from paconvert.global_var import GlobalManager
from paconvert.utils import log_info


class ImportTransformer(BaseTransformer):
    """
    Record import information
    """

    def __init__(
        self, root, file, imports_map, logger, all_api_map=None, unsupport_api_map=None
    ):
        super(ImportTransformer, self).__init__(
            root, file, imports_map, logger, all_api_map, unsupport_api_map
        )
        self.imports_map[self.file]["other_packages"] = set()
        self.imports_map[self.file]["torch_packages"] = set()
        self.imports_map[self.file]["may_torch_packages"] = set()
        self.imports_map[self.file]["api_alias_name_map"] = {}
        self.insert_pass_node = set()

    def visit_Import(self, node):
        """
        1. remove import torch.nn
        2. remove import torch.nn as nn
        3. record whether to import paddle
        4: 'import audiotools' -> 'import paddlespeech.audiotools as audiotools'
        5: 'import audiotools.ml' -> 'import paddlespeech.audiotools as audiotools'
        6: 'import audiotools as tools' -> 'import paddlespeech.audiotools as tools'
        7: 'import audiotools.ml as ml' -> 'import paddlespeech.audiotools.ml as ml'
        """
        new_node_names = []
        for alias_node in node.names:
            has_done = False

            # 1. import from current project
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
                break

            # 2. import from TORCH_PACKAGE_MAPPING (which means replace api one by one)
            for pkg_name in (
                list(GlobalManager.TORCH_PACKAGE_MAPPING.keys())
                + GlobalManager.MAY_TORCH_PACKAGE_LIST
            ):
                if (
                    alias_node.name.startswith(f"{pkg_name}.")
                    or pkg_name == alias_node.name
                ):
                    if pkg_name in GlobalManager.MAY_TORCH_PACKAGE_LIST:
                        self.imports_map[self.file]["may_torch_packages"].add(pkg_name)
                    else:
                        self.imports_map[self.file]["torch_packages"].add(pkg_name)
                    if alias_node.asname:
                        self.imports_map[self.file][alias_node.asname] = alias_node.name
                        log_info(
                            self.logger,
                            "remove 'import {} as {}' ".format(
                                alias_node.name, alias_node.asname
                            ),
                            self.file_name,
                            node.lineno,
                        )
                    else:
                        self.imports_map[self.file][alias_node.name] = alias_node.name
                        log_info(
                            self.logger,
                            "remove 'import {}' ".format(alias_node.name),
                            self.file_name,
                            node.lineno,
                        )
                    has_done = True
                    break
            if has_done:
                continue

            # 3. import form IMPORT_PACKAGE_MAPPING (which means replace api by all)
            for pkg_name in list(GlobalManager.IMPORT_PACKAGE_MAPPING.keys()):
                if f"{pkg_name}." in alias_node.name or pkg_name == alias_node.name:
                    replace_pkg_name = GlobalManager.IMPORT_PACKAGE_MAPPING[pkg_name]
                    if alias_node.asname:
                        # case 1: 'import audiotools as tools' -> 'import paddlespeech.audiotools as tools'
                        # case 2: 'import audiotools.ml as ml' -> 'import paddlespeech.audiotools.ml as ml'
                        replace_pkg_name = alias_node.name.replace(
                            pkg_name, replace_pkg_name
                        )
                        log_info(
                            self.logger,
                            "replace 'import {} as {}' to 'import {} as {}' ".format(
                                alias_node.name,
                                alias_node.asname,
                                replace_pkg_name,
                                alias_node.asname,
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        new_alias_node = ast.alias(replace_pkg_name, alias_node.asname)
                        new_node_names.append(new_alias_node)
                    else:
                        # case 1: 'import audiotools' -> 'import paddlespeech.audiotools as audiotools'
                        # case 2: 'import audiotools.ml' -> 'import paddlespeech.audiotools as audiotools'
                        log_info(
                            self.logger,
                            "replace 'import {}' to 'import {} as {}' ".format(
                                alias_node.name, replace_pkg_name, pkg_name
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        new_alias_node = ast.alias(replace_pkg_name, pkg_name)
                        new_node_names.append(new_alias_node)
                    has_done = True
                    break
            if has_done:
                continue

            # 4. import from other_packages
            self.insert_other_packages(self.imports_map, alias_node)
            new_node_names.append(alias_node)

        if len(new_node_names) > 0:
            node.names = new_node_names
            return node
        else:
            if (
                isinstance(self.parent_node, (ast.If, ast.Try))
                and self.parent_node not in self.insert_pass_node
            ):
                # case 1:
                # if cond:               ==> if cond:
                #   import numpy         ==>    import numpy
                #   import torch         ==>    pass
                # case 2:
                # if cond:               ==> if cond:
                #   import torch         ==>    pass
                #   import transformers  ==>
                # case 3:
                # if cond:               ==> if cond:
                #   import torch         ==>    pass
                #   import numpy         ==>    import numpy
                # case 4:
                # try:                   ==> try:
                #   import torch         ==>    pass
                # except:                ==> except:
                #   import numpy         ==>    import numpy
                self.insert_pass_node.add(self.parent_node)
                return ast.parse("pass").body[0]
            else:
                return None

    def visit_ImportFrom(self, node):
        """
        1. remove from torch import nn
        2. remove from torch import nn.functional as F
        3. 'from audiotools import AudioSignal' -> 'from paddlespeech.audiotools import AudioSignal'
        4. 'from audiotools.ml import BaseModel' -> 'from paddlespeech.audiotools.ml import BaseModel'
        5. 'from audiotools.ml import BaseModel as Model' -> 'from paddlespeech.audiotools.ml import BaseModel as Model'
        """
        # 1. import from current project
        if node.level > 0:
            # from ..datasets import xxx
            # from ... import xxx (node.module is None)

            """
            if "huggingface_internal" in self.file:
                # from ...configuration_utils import PretrainedConfig, layer_type_validation
                if node.level == 3:
                    if "pipeline_" in self.file:
                        node.module = ".".join(["diffusers", node.module])
                    else:
                        node.module = ".".join(["transformers", node.module])
                    node.level = 0
                elif node.level == 2:
                    if "pipeline_" in self.file:
                        node.module = ".".join(["diffusers.pipelines", node.module])
                    else:
                        node.module = ".".join(["transformers.models", node.module])
                    node.level = 0
                else:
                    self.insert_other_packages(self.imports_map, node)
                    return node
            else:
                self.insert_other_packages(self.imports_map, node)
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

        # 2. import from TORCH_PACKAGE_MAPPING (which means replace api one by one)
        for pkg_name in (
            list(GlobalManager.TORCH_PACKAGE_MAPPING.keys())
            + GlobalManager.MAY_TORCH_PACKAGE_LIST
        ):
            if node.module.startswith(f"{pkg_name}.") or pkg_name == node.module:
                if pkg_name in GlobalManager.MAY_TORCH_PACKAGE_LIST:
                    self.imports_map[self.file]["may_torch_packages"].add(pkg_name)
                else:
                    self.imports_map[self.file]["torch_packages"].add(pkg_name)
                for alias_node in node.names:
                    if alias_node.asname:
                        self.imports_map[self.file][alias_node.asname] = ".".join(
                            [node.module, alias_node.name]
                        )
                        log_info(
                            self.logger,
                            "remove 'from {} import {} as {}' ".format(
                                node.module, alias_node.name, alias_node.asname
                            ),
                            self.file_name,
                            node.lineno,
                        )
                    else:
                        self.imports_map[self.file][alias_node.name] = ".".join(
                            [node.module, alias_node.name]
                        )
                        log_info(
                            self.logger,
                            "remove 'from {} import {}' ".format(
                                node.module, alias_node.name
                            ),
                            self.file_name,
                            node.lineno,
                        )

                if (
                    isinstance(self.parent_node, (ast.If, ast.Try))
                    and self.parent_node not in self.insert_pass_node
                ):
                    # case 1:
                    # if cond:                    ==> if cond:
                    #   from torch import randn   ==>    pass
                    #   import numpy              ==>    import numpy
                    # case 2:
                    # if cond:                    ==> if cond:
                    #   from torch import randn   ==>    pass
                    #   from torch import matmul  ==>
                    # case 3:
                    # if cond:                    ==> if cond:
                    #   import numpy              ==>    import numpy
                    #   from torch import randn   ==>    pass
                    # case 4:
                    # try:                        ==> try:
                    #   from torch import randn   ==>    pass
                    # except:                     ==> except:
                    #   from numpy import randn   ==>    import numpy
                    self.insert_pass_node.add(self.parent_node)
                    return ast.parse("pass").body[0]
                else:
                    return None

        # 3. import form GlobalManager.IMPORT_PACKAGE_MAPPING (which means replace api by all)
        for pkg_name in list(GlobalManager.IMPORT_PACKAGE_MAPPING.keys()):
            if f"{pkg_name}." in node.module or pkg_name == node.module:
                # case 1: 'from audiotools import AudioSignal' -> 'from paddlespeech.audiotools import AudioSignal'
                # case 2: 'from audiotools.ml import BaseModel' -> 'from paddlespeech.audiotools.ml import BaseModel'
                # case 3: 'from audiotools.ml import BaseModel as Model' -> 'from paddlespeech.audiotools.ml import BaseModel as Model'
                origin_module = node.module
                node.module = origin_module.replace(
                    pkg_name, GlobalManager.IMPORT_PACKAGE_MAPPING[pkg_name]
                )
                log_info(
                    self.logger,
                    "replace 'from {} import xx' to 'from {} import xx' ".format(
                        origin_module, node.module
                    ),
                    self.file_name,
                    node.lineno,
                )
                return node

        # 4. other_packages
        self.insert_other_packages(self.imports_map, node)
        return node

    def insert_other_packages(self, imports_map, node):
        if isinstance(node, ast.ImportFrom):
            for alias_node in node.names:
                if alias_node.asname:
                    self.imports_map[self.file]["other_packages"].add(alias_node.asname)
                else:
                    # from data_loader.modules import *
                    if alias_node.name != "*":
                        self.imports_map[self.file]["other_packages"].add(
                            alias_node.name
                        )
        elif isinstance(node, ast.alias):
            if node.asname:
                self.imports_map[self.file]["other_packages"].add(node.asname)
            else:
                self.imports_map[self.file]["other_packages"].add(node.name)

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
            if torch_api in GlobalManager.ALIAS_MAPPING and (
                torch_api not in GlobalManager.NO_NEED_CONVERT_LIST
            ):
                torch_api = GlobalManager.ALIAS_MAPPING[torch_api]
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
            from torch.utils.cpp_extension import BuildExtension
            from transformers.activations import ACT2FN

            1. class A(Module):
            2. def func() -> Tensor:
            3. def func(x: Tensor):
            4. def func(dtype=float32):
            5. {'build_ext': BuildExtension}
            6. Tensor(2, 3)
            7. isinstance(x, Tensor)
            8. setattr(Tensor, 'add', func)
            9. inputs: Optional[Tensor] = None
            10. Union[GenerateOutput, torch.LongTensor]
            11. my_add = TorchAdd
            12. my_add(tensor_1,tensor_2)
            13. Union[List[str], List[AddedToken]],
            14. hasattr(Tensor, add)
            15. ACT2FN['tanh']
        """
        maybe_torch = False
        maybe_alias_name = False
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
            maybe_torch = True
        elif isinstance(self.parent_node, ast.Call) and isinstance(
            self.parent_node.func, ast.Name
        ):
            if self.parent_node.func == node:  # 6. Tensor(2, 3)
                maybe_torch = True
            if (
                node.id in self.imports_map[self.file]["api_alias_name_map"]
            ):  # 12. my_add(tensor_1,tensor_2)
                torch_api = self.imports_map[self.file]["api_alias_name_map"][node.id]
                return ast.parse(torch_api).body[0].value

            elif self.parent_node.func.id in [
                "isinstance",
                "setattr",
                "hasattr",
            ]:  # 7/8/14
                maybe_torch = True
        elif (
            isinstance(self.parent_node, ast.Subscript)
            and self.parent_node.slice == node
        ):
            maybe_torch = True  # 9. Optional[Tensor] = None
        elif (
            isinstance(self.parent_node, ast.Subscript)
            and self.parent_node.value == node
        ):
            # supplement api which can be indexed
            # from torch.utils import data
            # data[0]
            if node.id not in ["data"]:
                maybe_torch = True  # 15. ACT2FN['tanh']
        elif (
            isinstance(self.parent_node, ast.Tuple)
            and len(self.node_stack) >= 3
            and isinstance(self.node_stack[-3], ast.Subscript)
        ):
            maybe_torch = True  # 10. Union[GenerateOutput, torch.LongTensor]
        elif (
            isinstance(self.parent_node, ast.Assign) and node == self.parent_node.value
        ):
            maybe_torch = True  # 11. my_add = TorchAdd
            maybe_alias_name = True
        elif isinstance(self.parent_node, ast.Index) and self.parent_node.value == node:
            maybe_torch = True  # 13. Union[List[str], List[AddedToken]]

        if maybe_torch:
            torch_api = self.get_full_api_from_node(node)
            if torch_api:
                if torch_api in GlobalManager.ALIAS_MAPPING and (
                    torch_api not in GlobalManager.NO_NEED_CONVERT_LIST
                ):
                    torch_api = GlobalManager.ALIAS_MAPPING[torch_api]
                if maybe_alias_name:
                    if len(self.parent_node.targets) == 1 and isinstance(
                        self.parent_node.targets[0], ast.Name
                    ):
                        self.imports_map[self.file]["api_alias_name_map"][
                            self.parent_node.targets[0].id
                        ] = torch_api
                return ast.parse(torch_api).body[0].value
        return node

    def visit_Module(self, node):
        """
        'import torch_package' has been removed already, add 'import paddle_package'
        """
        super(ImportTransformer, self).generic_visit(node)
        line_NO = 1
        paddle_package_list = []
        for torch_package in self.imports_map[self.file]["torch_packages"]:
            paddle_package_list.append(
                GlobalManager.TORCH_PACKAGE_MAPPING[torch_package]
            )

        for may_torch_package in self.imports_map[self.file]["may_torch_packages"]:
            paddle_package_list.append(may_torch_package)

        for paddle_package in paddle_package_list:
            log_info(
                self.logger,
                f"add 'import {paddle_package}' in line {line_NO}",
                self.file_name,
            )
            self.record_scope(
                (self.root, "body", 0), ast.parse(f"import {paddle_package}").body
            )
            line_NO += 1
