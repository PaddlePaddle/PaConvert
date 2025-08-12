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
"""
   isort:skip_file
"""

import ast
import re

from paconvert.api_matcher import *
from paconvert.global_var import GlobalManager
from paconvert.base import BaseTransformer

from paconvert.utils import log_debug, log_info


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


class BasicTransformer(BaseTransformer):
    def __init__(
        self, root, file, imports_map, logger, all_api_map=None, unsupport_api_map=None
    ):
        super(BasicTransformer, self).__init__(
            root, file, imports_map, logger, all_api_map, unsupport_api_map
        )
        # use to identify tensor method/attribute
        self.black_list = list(self.imports_map[self.file]["other_packages"]) + [
            "ndarray",
            "args",
            "arg",
        ]

    def visit_Attribute(self, node):
        """
        torch api is not used by function call, so only match api name and not need to handle params.
        """
        # 1. torch.abs(x).transpose(1, 0)
        # 2. (x == y).transpose(1, 0)
        # 3. (x + y).transpose(1, 0)
        # 4. (-x).transpose(1, 0)
        # 5. x[0].transpose(1, 0)
        # 6. (x if flag else y).transpose(1, 0)
        if isinstance(
            node.value,
            (
                ast.Call,
                ast.Compare,
                ast.BinOp,
                ast.UnaryOp,
                ast.Subscript,
                ast.Assert,
                ast.IfExp,
            ),
        ):
            super(BasicTransformer, self).generic_visit(node)

        # 6. torch.tensor(features_A).T.cuda()
        if isinstance(node.value, ast.Attribute):
            if node.value.attr in [
                "T",
                "real",
                "weight",
                "bias",
                "imag",
            ]:
                super(BasicTransformer, self).generic_visit(node)
            # 7.  x.data.cuda()  / avoid  torch.utils.data.*
            elif (
                node.value.attr == "data"
                and "torch.utils" not in self.get_full_attr_for_apiname(node.value)
            ):
                super(BasicTransformer, self).generic_visit(node)

        # should be handled by visit_Call
        if isinstance(self.parent_node, ast.Call):
            if node == self.parent_node.func:
                return node

        # only need to convert:
        #   1. x.device...
        #   2. torch.Tensor/torch.nn.Module/torch.add...
        full_attr = self.get_full_attr_for_apiname(node)

        # 1) Torch Package Attribute, include torch third_party
        #   such as torch.Tensor/torch.nn.Module/torch.add...
        if self.start_with_torch(full_attr):
            torch_api = full_attr

            self.torch_api_count += 1
            if torch_api not in self.all_api_map:
                self.all_api_map[torch_api]["paddle_api"] = ""
                self.all_api_map[torch_api]["count"] = 0
            self.all_api_map[torch_api]["count"] += 1
            log_debug(
                self.logger,
                "Start convert {} to Paddle --> ".format(torch_api),
                self.file_name,
                node.lineno,
            )

            matcher = self.get_api_matcher(torch_api)
            # can be api_matcher or attribute_matcher
            if matcher is None:
                matcher = self.get_attribute_mather(torch_api)

            if matcher:
                paddle_api = matcher.get_paddle_api()
                if paddle_api == "delete":
                    if isinstance(self.parent_node, ast.Expr):
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Delete] Just remove {} ".format(torch_api),
                            self.file_name,
                            node.lineno,
                        )
                        return None
                    elif (
                        isinstance(self.parent_node, ast.FunctionDef)
                        and node in self.parent_node.decorator_list
                    ):
                        self.parent_node.decorator_list.remove(node)
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Delete] Just remove decorator",
                            self.file_name,
                            node.lineno,
                        )
                        return None
                elif paddle_api == "misidentify":
                    # This API usage indicate that is is not a Pytorch API
                    self.torch_api_count -= 1
                    del self.all_api_map[torch_api]
                    log_debug(
                        self.logger,
                        " Misidentify {}".format(torch_api),
                        self.file_name,
                        node.lineno,
                    )
                    return node
                elif paddle_api:
                    self.all_api_map[torch_api]["paddle_api"] = (
                        paddle_api if paddle_api else ""
                    )
                    new_node = ast.parse(paddle_api).body[0].value
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "[Success] Convert {} to Paddle".format(torch_api),
                        self.file_name,
                        node.lineno,
                    )
                    return new_node

            attr_list = full_attr.split(".")
            if len(attr_list) >= 3:
                # def add_module(self, module):
                #     ...
                # torch.nn.Module.add = add_module
                matcher = self.get_api_matcher(".".join(attr_list[:-1]))
                if matcher:
                    torch_api = ".".join(attr_list[:-1])
                    paddle_api = matcher.get_paddle_api()
                    new_node = ast.parse(paddle_api + "." + attr_list[-1]).body[0].value
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "[Success] Convert setattr({}, '{}') to Paddle".format(
                            torch_api, attr_list[-1]
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return new_node

            self.unsupport_api_map[torch_api] += 1
            log_info(
                self.logger,
                "[Not Support] Convert {} to Paddle is not supported currently".format(
                    torch_api
                ),
                self.file_name,
                node.lineno,
            )
            return node

        # 2) Torch Class attribute
        #   such as x.device...
        if "NonTorchClass" not in full_attr:
            is_tensor_api = False
            is_func_ctx_api = False
            is_distribution_api = False

            # when len(attr_list)> 2, need to more strict
            attr_list = full_attr.split(".")
            if len(attr_list) > 2:
                if "self." in full_attr:
                    # can be owned by other class
                    # self.weight.device
                    is_tensor_api = True

                for key in [".T.", ".data.", ".real.", ".imag.", ".weight.", ".bias."]:
                    if key in full_attr:
                        # x.T.device
                        # x.data.device
                        # x.real.device
                        # x.imag.device
                        # module.weight.device
                        # module.bias.device
                        is_tensor_api = True

            elif len(attr_list) == 2:
                if "self." in full_attr:
                    # can be inherit by users
                    pass
                else:
                    # Standard form
                    is_tensor_api = True
                    is_func_ctx_api = True
                    is_distribution_api = True

            torch_class_apis = []
            if is_tensor_api:
                torch_class_apis.append(".".join(["torch.Tensor", attr_list[-1]]))
            if is_func_ctx_api:
                torch_class_apis.append(
                    ".".join(["torch.autograd.function.FunctionCtx", attr_list[-1]])
                )
            if is_distribution_api:
                # config.mode
                if full_attr != "config.mode":
                    pass
                    # torch_class_apis.append(".".join(["paddle.Tensor", attr_list[-1]]))

            for torch_class_api in torch_class_apis:
                if self.in_attribute_mapping(torch_class_api):
                    self.torch_api_count += 1
                    if torch_class_api not in self.all_api_map:
                        self.all_api_map[torch_class_api]["paddle_api"] = ""
                        self.all_api_map[torch_class_api]["count"] = 0
                    self.all_api_map[torch_class_api]["count"] += 1
                    log_debug(
                        self.logger,
                        "Start convert Class Attribute: {} to Paddle --> ".format(
                            torch_class_api
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return self.trans_class_attribute(node, torch_class_api)

        # 3) Others
        return node

    def trans_class_attribute(self, node, torch_api):
        matcher = self.get_attribute_mather(torch_api)
        if matcher:
            self.all_api_map[torch_api]["paddle_api"] = (
                matcher.get_paddle_api() if matcher.get_paddle_api() else ""
            )
            node_list = matcher.get_paddle_class_attribute_nodes(node)
            if node_list == "delete":
                if isinstance(self.parent_node, ast.Expr):
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "[Delete] Just remove Class Attribute: {} ".format(torch_api),
                        self.file_name,
                        node.lineno,
                    )
                    return None
            elif node_list == "unchange":
                self.success_api_count += 1
                log_info(
                    self.logger,
                    "[Success] Convert Class Attribute: {} to Paddle, just remain the same".format(
                        torch_api
                    ),
                    self.file_name,
                    node.lineno,
                )
                return node
            elif node_list == "misidentify":
                # This API usage indicate that it is not this class attribute
                self.torch_api_count -= 1
                del self.all_api_map[torch_api]
                log_debug(
                    self.logger,
                    " Misidentify Class Attribute: {}".format(torch_api),
                    self.file_name,
                    node.lineno,
                )
                return node
            elif node_list:
                new_node = node_list[-1]
                if isinstance(new_node, ast.Expr):
                    new_node = new_node.value

                if isinstance(
                    new_node,
                    (
                        ast.Call,
                        ast.Attribute,
                        ast.Name,
                        ast.Constant,
                        ast.Compare,
                        ast.BinOp,
                        ast.UnaryOp,
                        ast.Tuple,
                        ast.Assert,
                    ),
                ):
                    if self.insert_multi_node(node_list[0:-1]):
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Success] Convert Class Attribute: {} to Paddle".format(
                                torch_api
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        return new_node

        annotate_node = ast.parse(
            "'Not Support auto convert {}, please judge whether it is Pytorch API and convert by yourself'".format(
                torch_api
            )
        ).body[0]
        self.record_scope(self.scope_body_index(), annotate_node)
        self.unsupport_api_map[torch_api] += 1
        log_info(
            self.logger,
            "[Not Support] convert Class Attribute: {} to Paddle is not supported currently".format(
                torch_api
            ),
            self.file_name,
            node.lineno,
        )
        return node

    def visit_Call(self, node):
        """
        if one line has N torch function, it has 2^N method of
        torch api and Tensor api Permutation and combination.

        eg:
        1. torch.reshape(torch.add(torch.abs(x), y), [3])  :  3 torch api
        ast.Call
        -> [func]ast.Attribute + [args]ast.Call
                                  -> [func]ast.Attribute + [args]ast.Call(torch.abs)

        2. torch.reshape(torch.add(x.abs(), y), [3])  :  2 torch api + 1 tensor api
        ast.Call
        -> [func]ast.Attribute + [args]ast.Call
                                  -> [func]ast.Attribute + [args]ast.Call(x.abs)

        3. torch.reshape(torch.abs(x).add(y), [3])  :  2 torch api + 1 tensor api
        ast.Call
        -> [func]ast.Attribute + [args]ast.Call
                                  -> [func]ast.Attribute([value]ast.Call)(torch.abs)

        4. torch.add(torch.abs(x), y).reshape([3])  :  2 torch api + 1 tensor api
        ast.Call
        -> [func]ast.Attribute([value]ast.Call)
                                -> [func]ast.Attribute + [args]ast.Call(torch.abs)

        5. torch.abs(x).add(y).reshape([3])  :  1 torch api + 2 tensor api
        ast.Call
        -> [func]ast.Attribute([value]ast.Call)
                                -> [func]ast.Attribute([value]ast.Call)(torch.abs)

        6. torch.add(x.abs(), y).reshape([3])  :  1 torch api + 2 tensor api
        ast.Call
        -> [func]ast.Attribute([value]ast.Call)
                                -> [func]ast.Attribute + [args]ast.Call(x.abs)

        7. torch.reshape(x.abs().add(y), [3])  :  1 torch api + 2 tensor api
        ast.Call
        -> [func]ast.Attribute + [args]ast.Call
                                  -> [func]ast.Attribute([value]ast.Call)(x.abs)

        8. x.abs().add(y).reshape([3])  :  3 tensor api
        ast.Call
        -> [func]ast.Attribute([value]ast.Call)
                                  -> [func]ast.Attribute([value]ast.Call)(x.abs)

        Therefore, 8 method: 2*2*2, each call has 2 kind call:
         - torch api: [args]ast.Call
         - tensor api: [func]ast.Attribute([value]ast.Call)
        """
        # Use Postorder traversal
        super(BasicTransformer, self).generic_visit(node)

        full_attr = self.get_full_attr_for_apiname(node.func)
        # 1) Torch Package Call, include torch third_party
        #   such as : torch.add(x, y) / torch.add(torch.abs(x), y)
        #   for may_torch_package_list, will in_api_mapping
        if self.start_with_torch(full_attr) or self.in_api_mapping(full_attr):
            torch_api = full_attr
            self.torch_api_count += 1
            if torch_api not in self.all_api_map:
                self.all_api_map[torch_api]["paddle_api"] = ""
                self.all_api_map[torch_api]["count"] = 0
            self.all_api_map[torch_api]["count"] += 1
            log_debug(
                self.logger,
                "Start convert {} to Paddle --> ".format(torch_api),
                self.file_name,
                node.lineno,
            )

            matcher = self.get_api_matcher(torch_api)
            if matcher:
                self.all_api_map[torch_api]["paddle_api"] = (
                    matcher.get_paddle_api() if matcher.get_paddle_api() else ""
                )
                node_list = matcher.get_paddle_nodes(node.args, node.keywords)
                if node_list == "delete":
                    if isinstance(self.parent_node, ast.Expr):
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[[Delete]] Just remove {} ".format(torch_api),
                            self.file_name,
                            node.lineno,
                        )
                        return None
                elif node_list == "misidentify":
                    # This API usage indicate that is is not a Pytorch API
                    self.torch_api_count -= 1
                    del self.all_api_map[torch_api]
                    log_debug(
                        self.logger,
                        " Misidentify {}".format(torch_api),
                        self.file_name,
                        node.lineno,
                    )
                    return node
                elif node_list:
                    new_node = node_list[-1]
                    # ast.Expr, which contain ast.Call or ast.Name
                    if isinstance(new_node, ast.Expr):
                        new_node = new_node.value

                    if isinstance(
                        new_node,
                        (
                            ast.Call,
                            ast.Name,
                            ast.Constant,
                            ast.Attribute,
                            ast.Compare,
                            ast.BinOp,
                            ast.UnaryOp,
                            ast.Tuple,
                            ast.Assert,
                            ast.Subscript,
                        ),
                    ):
                        if self.insert_multi_node(node_list[0:-1]):
                            self.success_api_count += 1
                            log_info(
                                self.logger,
                                "[Success] Convert {} to Paddle".format(torch_api),
                                self.file_name,
                                node.lineno,
                            )
                            return new_node

            self.unsupport_api_map[torch_api] += 1

            log_info(
                self.logger,
                "[Not Support] convert {} to Paddle is not supported currently".format(
                    torch_api
                ),
                self.file_name,
                node.lineno,
            )
            return node

        # 2) Torch Class call
        #   such as : x.add(y) / x.abs().add / sgd.step() / model.to(torch.device('cuda'))
        if "NonTorchClass" not in full_attr:
            is_tensor_api = False
            is_module_api = False
            is_optim_api = False
            is_func_ctx_api = False
            is_distribution_api = False
            is_profile_api = False
            is_auto_grad_profile_api = False

            # when len(attr_list)> 2, need to more strict
            attr_list = full_attr.split(".")
            if len(attr_list) > 2:
                if "self." in full_attr:
                    # can be owned by other class
                    # self.weight.add
                    # self.lienar.add_module
                    # self.optimizer.load_state_dict
                    is_tensor_api = True
                    is_module_api = True
                    is_optim_api = True

                if (
                    ".T." in full_attr
                    or ".weight." in full_attr
                    or ".bias." in full_attr
                ):
                    # x.T.zero_()
                    # x.weight.zero_()
                    # x.bias.zero_()
                    is_tensor_api = True

            elif len(attr_list) == 2:
                if "self." in full_attr:
                    # can be inherit by users
                    # self.add_module
                    # self.load_state_dict
                    is_module_api = True
                    is_optim_api = True
                else:
                    # Standard form
                    is_tensor_api = True
                    is_module_api = True
                    is_optim_api = True
                    is_func_ctx_api = True
                    is_distribution_api = True
                    is_profile_api = True
                    is_auto_grad_profile_api = True

            torch_class_apis = []
            if is_tensor_api:
                torch_class_apis.append(".".join(["torch.Tensor", attr_list[-1]]))
            if is_module_api:
                torch_class_apis.append(".".join(["torch.nn.Module", attr_list[-1]]))
            if is_optim_api:
                torch_class_apis.append(
                    ".".join(["torch.optim.Optimizer", attr_list[-1]])
                )
            if is_func_ctx_api:
                torch_class_apis.append(
                    ".".join(["torch.autograd.function.FunctionCtx", attr_list[-1]])
                )
            if is_distribution_api:
                torch_class_apis.append(
                    ".".join(
                        ["torch.distributions.distribution.Distribution", attr_list[-1]]
                    )
                )
            if is_profile_api:
                torch_class_apis.append(
                    ".".join(["torch.profiler.profile", attr_list[-1]])
                )
            if is_auto_grad_profile_api:
                torch_class_apis.append(
                    ".".join(["torch.autograd.profiler.profile", attr_list[-1]])
                )

            for torch_class_api in torch_class_apis:
                if self.in_api_mapping(torch_class_api):
                    self.torch_api_count += 1
                    if torch_class_api not in self.all_api_map:
                        self.all_api_map[torch_class_api]["paddle_api"] = ""
                        self.all_api_map[torch_class_api]["count"] = 0
                    self.all_api_map[torch_class_api]["count"] += 1
                    log_debug(
                        self.logger,
                        "Start convert Class Method: {} to Paddle --> ".format(
                            torch_class_api
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return self.trans_class_method(node, torch_class_api)

        # 3) Others
        return node

    def trans_class_method(self, node, torch_api):
        matcher = self.get_api_matcher(torch_api)
        if matcher:
            node_args = node.args
            # static method call
            # PT_Optimizer.load_state_dict(self, swa_state_dict)
            self_in_args = False
            if len(node_args) >= 1:
                if isinstance(node_args[0], ast.Name) and node_args[0].id == "self":
                    self_in_args = True
                    node_args = node_args[1:]

            self.all_api_map[torch_api]["paddle_api"] = (
                matcher.get_paddle_api() if matcher.get_paddle_api() else ""
            )
            node_list = matcher.get_paddle_class_nodes(
                node.func, node_args, node.keywords
            )
            if node_list == "delete":
                if isinstance(self.parent_node, ast.Expr):
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "[Delete] Just remove Class Method: {} ".format(torch_api),
                        self.file_name,
                        node.lineno,
                    )
                    return None
            elif node_list == "unchange":
                self.success_api_count += 1
                log_info(
                    self.logger,
                    "[Success] Convert Class Method: {} to Paddle, just remain the same".format(
                        torch_api
                    ),
                    self.file_name,
                    node.lineno,
                )
                return node
            elif node_list == "misidentify":
                self.torch_api_count -= 1
                del self.all_api_map[torch_api]
                # This API usage indicate that it is not this class method
                log_debug(
                    self.logger,
                    " Misidentify Class Method: {}".format(torch_api),
                    self.file_name,
                    node.lineno,
                )
                return node
            elif node_list:
                new_node = node_list[-1]
                # ast.Expr which contain ast.Call or ast.Name
                if isinstance(new_node, ast.Expr):
                    new_node = new_node.value

                # return value can be:
                #   x.abs()
                #   x
                #   'float32'
                #   x.shape
                #   x.shape[2]
                #   x * 2
                #   assert x=1.
                #   (x, y)
                #   x > 1
                if isinstance(
                    new_node,
                    (
                        ast.Call,
                        ast.Name,
                        ast.Constant,
                        ast.Attribute,
                        ast.Subscript,
                        ast.BinOp,
                        ast.Assert,
                        ast.Tuple,
                        ast.Compare,
                    ),
                ):
                    if self_in_args:
                        if isinstance(new_node, ast.Call):
                            new_node.args.insert(0, ast.Name(id="self", ctx=ast.Load()))

                    if self.insert_multi_node(node_list[0:-1]):
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Success] Convert Class Method: {} to Paddle".format(
                                torch_api
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        return new_node

        torch_api = "*" + torch_api[torch_api.rfind(".") :]
        annotate_node = ast.parse(
            "'Not Support auto convert {}, please judge whether it is Pytorch API and convert by yourself'".format(
                torch_api
            )
        ).body[0]
        self.record_scope(self.scope_body_index(), annotate_node)
        self.unsupport_api_map[torch_api] += 1
        log_info(
            self.logger,
            "[Not Support] convert Class Method: {} to Paddle is not supported currently".format(
                torch_api
            ),
            self.file_name,
            node.lineno,
        )
        return node

    def start_with_torch(self, torch_api):
        for torch_package in self.imports_map[self.file]["torch_packages"]:
            if torch_api.startswith("%s." % torch_package):
                return True

        return False

    def in_api_mapping(self, torch_api):
        if torch_api in GlobalManager.NO_NEED_CONVERT_LIST:
            return True
        if torch_api in GlobalManager.ALIAS_MAPPING:
            torch_api = GlobalManager.ALIAS_MAPPING[torch_api]
        if torch_api in GlobalManager.API_MAPPING:
            return True
        for wildcard_name in list(GlobalManager.API_WILDCARD_MAPPING.keys()):
            if re.match(wildcard_name, torch_api):
                return True
        return False

    def get_api_matcher(self, torch_api):
        api_mapping_dict = {}
        if torch_api in GlobalManager.NO_NEED_CONVERT_LIST:
            return NoNeedConvertMatcher(self, torch_api, {}, self.logger)

        if torch_api in GlobalManager.ALIAS_MAPPING:
            torch_api = GlobalManager.ALIAS_MAPPING[torch_api]
        if torch_api in GlobalManager.API_MAPPING:
            api_mapping_dict = GlobalManager.API_MAPPING[torch_api]
        else:
            for wildcard_name in list(GlobalManager.API_WILDCARD_MAPPING.keys()):
                if re.match(wildcard_name, torch_api):
                    api_mapping_dict = GlobalManager.API_WILDCARD_MAPPING[wildcard_name]

        if api_mapping_dict:
            if "disable" in api_mapping_dict:
                return None
            if "Matcher" in api_mapping_dict:
                matcher = api_mapping_dict["Matcher"]
                return eval(matcher)(self, torch_api, api_mapping_dict, self.logger)
        return None

    def in_attribute_mapping(self, torch_api):
        if torch_api in GlobalManager.NO_NEED_CONVERT_LIST:
            return True
        if torch_api in GlobalManager.ALIAS_MAPPING:
            torch_api = GlobalManager.ALIAS_MAPPING[torch_api]
        if torch_api in GlobalManager.ATTRIBUTE_MAPPING:
            return True
        return False

    def get_attribute_mather(self, torch_api):
        attr_mapping_dict = {}
        if torch_api in GlobalManager.NO_NEED_CONVERT_LIST:
            return NoNeedConvertMatcher(self, torch_api, {}, self.logger)

        if torch_api in GlobalManager.ALIAS_MAPPING:
            torch_api = GlobalManager.ALIAS_MAPPING[torch_api]
        if torch_api in GlobalManager.ATTRIBUTE_MAPPING:
            attr_mapping_dict = GlobalManager.ATTRIBUTE_MAPPING[torch_api]
        else:
            for wildcard_name in list(GlobalManager.API_WILDCARD_MAPPING.keys()):
                if re.match(wildcard_name, torch_api):
                    attr_mapping_dict = GlobalManager.API_WILDCARD_MAPPING[
                        wildcard_name
                    ]

        if attr_mapping_dict:
            if "disable" in attr_mapping_dict:
                return None
            if "Matcher" in attr_mapping_dict:
                matcher = attr_mapping_dict["Matcher"]
                return eval(matcher)(self, torch_api, attr_mapping_dict, self.logger)
        return None

    def visit_Expr(self, node):
        for field, old_value in iter_fields(node):
            new_node = self.visit(old_value)
            if new_node is None:
                return None
            else:
                setattr(node, field, new_node)
        return node

    def visit_Name(self, node):
        """
        The torch api is a torch-related package name, rather than an attribute.

        eg.
            import torch
            import transformers

            class nn_mymodule():
                ...

            1. setattr(torch,"nn", nn_mymodule)
            2. hasattr(torch, "nn")
            3. hasattr(transformers,"__version__")
        """
        if (
            isinstance(self.parent_node, ast.Call)
            and isinstance(self.parent_node.func, ast.Name)
            and self.parent_node.func.id
            in [
                "setattr",
                "hasattr",
            ]
        ):
            if node.id in GlobalManager.TORCH_PACKAGE_MAPPING:
                return (
                    ast.parse(GlobalManager.TORCH_PACKAGE_MAPPING[node.id])
                    .body[0]
                    .value
                )
        return node
