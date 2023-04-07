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

import json
import os
import ast
from os import path
import collections

import sys
sys.path.append(path.dirname(__file__)+"../")

from paconvert.api_matcher import *
from paconvert.base import API_MAPPING, ATTRIBUTE_MAPPING, TORCH_PACKAGE_LIST, BaseTransformer

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
    def __init__(self, root, file, imports_map, logger, unsupport_map=None):
        super(BasicTransformer, self).__init__(root, file, imports_map, logger)
        # use to identify tensor method/attribute
        self.black_list = self.imports_map[self.file]['other_pacakages'] + ['ndarray', 'args', 'arg']
        self.unsupport_map = unsupport_map

    @property
    def parent_node(self):
        return self.node_stack[-2]
        
    def scope_node_body_index(self, level=-1):
        scope_node = self.scope_stack[level]

        # reverse find scope_node in node_stack
        lower = -1 * (len(self.node_stack) + 1)
        for i in range(-1, lower, -1):
            if self.node_stack[i] == scope_node:
                for index, node in enumerate(scope_node.body):
                    if node == self.node_stack[i+1]:
                        return scope_node, 'body', index
                
                if getattr(scope_node, 'orelse', None):
                    for index, node in enumerate(scope_node.orelse):
                        if node == self.node_stack[i+1]:
                            return scope_node, 'orelse', index

                if getattr(scope_node, 'decorator_list', None):
                    for index, node in enumerate(scope_node.decorator_list):
                        if node == self.node_stack[i+1]:
                            return scope_node, 'decorator_list', index

        return self.scope_node_body_index(-2)

    def visit_Attribute(self, node):
        '''
        torch api is not used by funcition call, so only match api name and not need to handle params. 
        '''
        # 1. torch.abs(x).transpose(1, 0)
        # 2. (x == y).transpose(1, 0)
        # 3. (x + y).transpose(1, 0)
        # 4. (-x).transpose(1, 0)
        # 5. x[0].transpose(1, 0)
        if isinstance(node.value, (ast.Call, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Subscript)):
            super(BasicTransformer, self).generic_visit(node)
        
        # 6.torch.tensor(features_A).T.cuda()
        if isinstance(node.value, ast.Attribute) and node.value.attr == 'T':
            super(BasicTransformer, self).generic_visit(node)

        # should be handled by visit_Call
        if isinstance(self.parent_node, ast.Call):
            if node == self.parent_node.func:
                return node
        
        # only need to convert:
        #   1. x.device...
        #   2. torch.Tensor/torch.nn.Module/torch.add...
        full_attr = self.get_full_attr(node)

        # Torch Package Attribute, include torch third_party
        #   such as torch.Tensor/torch.nn.Module/torch.add...
        for torch_package in TORCH_PACKAGE_LIST:
            if full_attr.startswith("%s." % torch_package):
                torch_api = full_attr
                self.torch_api_count += 1
                self.log_debug("Start convert {} to Paddle --> ".format(torch_api), self.file_name, node.lineno)
                matcher = self.get_api_mather(torch_api)
                if matcher:
                    paddle_api = matcher.get_paddle_api()
                    if paddle_api == 'delete':
                        if isinstance(self.parent_node, ast.Expr):
                            self.success_api_count += 1
                            self.log_debug("[Success]remove {} ".format(torch_api), self.file_name, node.lineno)
                            return None
                    elif paddle_api:
                        new_node = ast.parse(paddle_api).body[0].value
                        self.success_api_count += 1
                        self.log_debug("[Success] convert {} to Paddle Successfully".format(torch_api), self.file_name, node.lineno)
                        return new_node
                
                self.unsupport_map[torch_api] += 1
                self.log_info("[Not Support] convert {} to Paddle is not supported currently".format(torch_api), self.file_name, node.lineno)
                return node

        # Torch Class attribute, such as: x.device
        #   such as x.device...
        if 'NonTorchClass' not in full_attr:
            attr_list = full_attr.split('.')
            torch_api = '.'.join(['torch.Tensor', attr_list[-1]])
            if torch_api in ATTRIBUTE_MAPPING:
                self.torch_api_count += 1
                self.log_debug("Start convert Tensor Attribute: {} to Paddle ".format(torch_api), self.file_name, node.lineno)
                return self.trans_class_attribute(node, torch_api)

        # NonTorchClass
        return node

    def trans_class_attribute(self, node, torch_api):
        if torch_api in ATTRIBUTE_MAPPING:
            attribute_mapping = ATTRIBUTE_MAPPING[torch_api]
            if "Matcher" in attribute_mapping:
                matcher = eval(attribute_mapping['Matcher'])(torch_api, attribute_mapping)
                if matcher:
                    paddle_api = matcher.get_paddle_api()
                    if paddle_api:
                        self.success_api_count += 1
                        self.log_debug("[Success]convert Tensor Attribute: {} to Paddle".format(torch_api), self.file_name, node.lineno)
                        node.attr = ast.parse(paddle_api).body[0].value.attr
                        if paddle_api == 'paddle.Tensor.stop_gradient':
                            if isinstance(self.parent_node, (ast.Tuple)):
                                temp_index = -1
                                for index in range(len(self.root.body)):
                                    if isinstance(self.root.body[index], (ast.Assign)) and self.parent_node in self.root.body[index].targets:
                                        temp_index = index
                                if temp_index != -1:
                                    new_node = ast.Name(id='temp', ctx=ast.Load())
                                    assign_node = ast.Assign(targets=[node], value = ast.UnaryOp(ast.Not(), operand = new_node))
                                    self.root.body.insert(temp_index+1, assign_node)
                                    return new_node
                                else:
                                    node = ast.UnaryOp(ast.Not(), operand = node)
                            elif isinstance(self.parent_node, (ast.Assign)) and node == self.parent_node.value:
                                node = ast.UnaryOp(ast.Not(), operand = node)
                            elif isinstance(self.parent_node, (ast.Assign, ast.keyword)):
                                self.parent_node.value = ast.UnaryOp(ast.Not(), operand = self.parent_node.value)
                            else:
                                node = ast.UnaryOp(ast.Not(), operand = node)
                            return node                                
                        # for tensor attribute , only need to change .attr
                        return node

        annotate_node = ast.parse("'Tensor Attribute: {}, not convert, please check whether it is torch.Tensor.* and convert manually'".format(torch_api)).body[0]
        self.record_scope(self.scope_node_body_index(), annotate_node)
        self.unsupport_map[torch_api] += 1
        self.log_info("[Not Support] convert Tensor Attribute: {} to Paddle is not supported currently".format(torch_api), self.file_name, node.lineno)
        return node


    def visit_Call(self, node):
        '''
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
        '''
        full_attr = self.get_full_attr(node.func)

        # Use Postorder traversal
        super(BasicTransformer, self).generic_visit(node)
        
        # Torch Package Call, include torch third_party
        #   such as : torch.add(x, y) / torch.add(torch.abs(x), y)
        for torch_package in TORCH_PACKAGE_LIST:
            if full_attr.startswith("%s." % torch_package):
                torch_api = full_attr
                self.torch_api_count += 1
                self.log_debug("Start convert {} to Paddle --> ".format(torch_api), self.file_name, node.lineno)
                support = True

                matcher = self.get_api_mather(torch_api)
                if not matcher:
                    support = False
                # such as torch.max(*args, **kwargs)
                if isinstance(node.args, ast.Starred):
                    support = False
                for k_node in node.keywords:
                    if k_node.arg is None:
                        support = False
                        
                if support:
                    node_list = matcher.get_paddle_nodes(node.args, node.keywords)
                    if node_list == 'delete':
                        if isinstance(self.parent_node, ast.Expr):
                            self.success_api_count += 1
                            self.log_debug("[Success]remove {} ".format(torch_api), self.file_name, node.lineno)
                            return None
                    elif node_list:
                        new_node = node_list[-1]
                        # ast.Expr, which contain ast.Call or ast.Name
                        if isinstance(new_node, ast.Expr):
                            new_node = new_node.value
                        
                        if isinstance(new_node, (ast.Call, ast.Name, ast.Constant, ast.Attribute, ast.Compare)):
                            # if multiple line, record lines and will insert after all node visit
                            if node_list[0:-1]:
                                self.log_debug("insert extra {} lines for torch api {}".format(len(node_list[0:-1]), torch_api), self.file_name, node.lineno)
                                self.record_scope(self.scope_node_body_index(), node_list[0:-1])

                            self.success_api_count += 1
                            self.log_debug("[Success]convert {} to Paddle ".format(torch_api), self.file_name, node.lineno)
                            return new_node

                self.unsupport_map[torch_api] += 1
                self.log_info("[Not Support] convert {} to Paddle is not supported currently".format(torch_api), self.file_name, node.lineno)
                return node


        # Torch Class call
        #   such as : x.add(y) / x.abs().add / sgd.step() / model.to(torch.device('cuda'))
        if 'NonTorchClass' not in full_attr:
            attr_list = full_attr.split('.')
            #  x.reshape
            #  self.weight.reshape
            #  x.T.reshape
            # when > 2, need to more strict
            if (len(attr_list) == 2 and 'self' not in full_attr) or (len(attr_list) > 2 and 'self' in full_attr) or '.T.' in full_attr:
                torch_api = '.'.join(['torch.Tensor', attr_list[-1]])
                if torch_api in API_MAPPING:
                    self.torch_api_count += 1
                    self.log_debug("Start convert Tensor Class Method: {} to Paddle --> ".format(torch_api), self.file_name, node.lineno)
                    return self.trans_class_method(node, torch_api)

                torch_api = '.'.join(['torch.nn.Module', attr_list[-1]])
                if torch_api in API_MAPPING:
                    self.torch_api_count += 1
                    self.log_debug("Start convert Layer Class Method: {} to Paddle --> ".format(torch_api), self.file_name, node.lineno)
                    return self.trans_class_method(node, torch_api)

                torch_api = '.'.join(['torch.optim.Optimizer', attr_list[-1]])
                if torch_api in API_MAPPING:
                    self.torch_api_count += 1
                    self.log_debug("Start convert Optimizer Class Method: {} to Paddle --> ".format(torch_api), self.file_name, node.lineno)
                    return self.trans_class_method(node, torch_api)

        # NonTorchClass
        return node


    def trans_class_method(self, node, torch_api):
        matcher = self.get_api_mather(torch_api)
        if matcher:
            node_list = matcher.get_paddle_class_nodes(node.func, node.args, node.keywords)
            if node_list == "NonTorchClass":
                # This API usage  indicate that is is not a torch.Tensor
                self.torch_api_count -= 1
                self.log_debug(" Misidentify Class Method: {}, so just remain the same".format(torch_api), self.file_name, node.lineno)
                return node
            elif node_list is not None:
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
                if isinstance(new_node, (ast.Call, ast.Name, ast.Constant, ast.Attribute, ast.Subscript)):
                    # if multiple line, record lines and will insert after all node visit
                    if node_list[0: -1]:
                        self.log_debug("insert extra {} lines for Class Method: {}".format(len(node_list[0: -1]), torch_api), self.file_name, node.lineno)
                        self.record_scope(self.scope_node_body_index(), node_list[0: -1])

                    self.success_api_count += 1
                    self.log_debug("[Success]convert Class Method: {} to Paddle ".format(torch_api), self.file_name, node.lineno)
                    return new_node
        
        torch_api = '*' + torch_api[torch_api.rfind('.'): ]
        annotate_node = ast.parse("'Class Method: {}, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually'".format(torch_api)).body[0]
        self.record_scope(self.scope_node_body_index(), annotate_node)
        self.unsupport_map[torch_api] += 1
        self.log_info("[Not Support] convert Class Method: {} to Paddle is not supported currently".format(torch_api), self.file_name, node.lineno)
        return node


    def get_api_mather(self, torch_api):
        if torch_api in API_MAPPING:
            api_mapping = API_MAPPING[torch_api]
            if "disable" in api_mapping and eval(api_mapping["disable"]):
                return None

            if "Matcher" in api_mapping:
                matcher = api_mapping['Matcher']
                return eval(matcher)(torch_api, api_mapping)
        return None

    def visit_Expr(self, node):
        for field, old_value in iter_fields(node):
            new_node = self.visit(old_value)
            if new_node is None:
                return None
            else:
                setattr(node, field, new_node)
        return node

    def visit_FunctionDef(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node
    
    def visit_While(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_If(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Try(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_TryFinally(self, node):
        self.scope_stack.append(node)
        node = super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_For(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_With(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        return node

    def visit_Module(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
        #self.log_debug("Mark this file which has been converted already", self.file_name)
        #mark_node = ast.parse("' This file is generated by Paddle converter, you can remove this mark'").body[0]
        #self.record_scope((self.root, 'body', 0), mark_node)
        return node
