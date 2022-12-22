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

import sys
sys.path.append(path.dirname(__file__)+"../")

from paddleconverter.api_matcher import *
from paddleconverter.base import API_MAPPING, BaseTransformer

class BasicTransformer(BaseTransformer):
    @property
    def parent_node(self):
        return self.node_stack[-2]

    @property
    def scope_node(self):
        return self.scope_stack[-1]

    def scope_body_index(self):
        lower = -1 * (len(self.node_stack) + 1)
        for i in range(-1, lower, -1):
            if self.node_stack[i] == self.scope_node:
                for index, node in enumerate(self.scope_node.body):
                    if self.node_stack[i+1] == node:
                        return 'body', index
                
                if getattr(self.scope_node, 'orelse'):
                    for index, node in enumerate(self.scope_node.orelse):
                        if self.node_stack[i+1] == node:
                            return 'orelse', index
            
        return 'body', 0

    def visit_Attribute(self, node):
        '''
        torch api is not used by funcition call, so only match api name and not need to handle params. 
        '''
        # 1. torch.abs(x).transpose(1, 0)
        # 2. (x == y).transpose(1, 0)
        # 3. (x + y).transpose(1, 0)
        # 4. x[0].transpose(1, 0)
        # 5. (-x).transpose
        if isinstance(node.value, (ast.Call, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Subscript)):
            super(BasicTransformer, self).generic_visit(node)
            
        if isinstance(self.parent_node, ast.Call):
            if node == self.parent_node.func:
                return node
            
        full_attr = self.get_full_attr(node)
        
        # Tensor attribute, such as: x.device / x.dtype
        if not full_attr.startswith('torch.'):
            if not full_attr.startswith('None') and len(full_attr.split('.')) == 2:
                attr_list = full_attr.split('.')
                WHITE_LIST = self.imports_map[self.file]['others']
                WHITE_LIST += ['self', 'args', 'line', 'lines']
                # Avoid ' np.add, scipy.add ... '
                if attr_list[0] not in WHITE_LIST:
                    attr_list[0] = 'torch.Tensor'
                    if '.'.join(attr_list) in API_MAPPING:
                        torch_api = '.'.join(attr_list)
                        self.torch_api_count += 1
                        return self.trans_tensor_attribute(node, torch_api)

        # Non-Tensor attribute, such as: torch.device / torch.dtype
        if full_attr.startswith('torch.'):
            torch_api = full_attr
            self.torch_api_count += 1
            matcher = self.get_api_mather(torch_api)
            if matcher:
                paddle_api = matcher.get_paddle_api()
                if paddle_api:
                    self.success_api_count += 1
                    self.log_info("[Success]convert {} to Paddle".format(torch_api), self.file_name, node.lineno)
                    new_node = ast.parse(paddle_api).body[0].value
                    return new_node

            self.log_info("[Failed]can not convert {} to Paddle".format(torch_api), self.file_name, node.lineno)
        return node 

    def trans_tensor_attribute(self, node, torch_api):
        body, body_index = self.scope_body_index()
        matcher = self.get_api_mather(torch_api)
        if matcher:
            paddle_api = matcher.get_paddle_api()
            if paddle_api:
                self.success_api_count += 1
                self.log_info("[Success]convert Tensor Attribute {} to Paddle".format(torch_api), self.file_name, node.lineno)
                # for tensor attribute , only need to change .device
                node.attr = ast.parse(paddle_api).body[0].value.attr
                return node

        annotate_node = ast.parse("'Tensor Attribute: {}, not convert, please check whether to convert manually'".format(torch_api)).body[0]
        self.record_scope(self.scope_node, (body, body_index), annotate_node)
        self.log_info("[Failed]can not convert Tensor Attribute {} to Paddle ".format(torch_api), self.file_name, node.lineno)
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
        
        # Use Postorder traversal
        super(BasicTransformer, self).generic_visit(node)
        
        full_attr = self.get_full_attr(node.func)
        
        # Tensor method func, such as : x.add(y) / x.abs().add
        if not full_attr.startswith('torch.'):
            if not full_attr.startswith('None') and len(full_attr.split('.')) == 2:
                attr_list = full_attr.split('.')
                WHITE_LIST = self.imports_map[self.file]['others']
                WHITE_LIST += ['self']
                # Avoid ' np.add, scipy.add ... '
                if attr_list[0] not in WHITE_LIST:
                    attr_list[0] = 'torch.Tensor'
                    if '.'.join(attr_list) in API_MAPPING:
                        torch_api = '.'.join(attr_list)
                        self.torch_api_count += 1
                        return self.trans_tensor_method(node, torch_api)
        
        # Non-Tensor method, such as : torch.add(x,y) / torch.add(torch.abs(x), y)
        if full_attr.startswith('torch.'):
            torch_api = full_attr
            self.torch_api_count += 1     

            matcher = self.get_api_mather(torch_api)
            if matcher:
                node_list = matcher.get_paddle_nodes(node.args, node.keywords)
                if node_list:
                    new_node = node_list[-1]
                    # ast.Expr, which contain ast.Call or ast.Name
                    if isinstance(new_node, ast.Expr):
                        new_node = new_node.value
                    
                    if isinstance(new_node, (ast.Call, ast.Name, ast.Constant, ast.Attribute)):
                        self.success_api_count += 1
                        self.log_info("[Success]convert {} to Paddle ".format(torch_api), self.file_name, node.lineno)
                        
                        # if multiple line, record lines and will insert after all node visit
                        if node_list[0:-1]:
                            self.log_info("insert extra {} lines for torch api {}".format(len(node_list[0:-1]), torch_api), self.file_name, node.lineno)
                            self.record_scope(self.scope_node, self.scope_body_index(), node_list[0:-1])

                        return new_node

            self.log_info("[Failed]can not convert {} to Paddle ".format(torch_api), self.file_name, node.lineno)
        
        return node 


    def trans_tensor_method(self, node, torch_api):
        body, body_index = self.scope_body_index()
        matcher = self.get_api_mather(torch_api)
        if matcher:
            node_list = matcher.get_paddle_tensor_nodes(node.func, node.args, node.keywords)
            if node_list:
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
                    self.success_api_count += 1
                    self.log_info("[Success]convert Tensor Method {} to Paddle ".format(torch_api), self.file_name, node.lineno)

                    # if multiple line, record lines and will insert after all node visit
                    if node_list[0:-1]:
                        self.log_info("insert extra {} lines for torch api {}".format(len(node_list[0:-1]), torch_api), self.file_name, node.lineno)
                        self.record_scope(self.scope_node, (body, body_index), node_list[0:-1])

                    return new_node

        annotate_node = ast.parse("'Tensor Method: {}, can not convert, please check whether to convert manually'".format(torch_api)).body[0]
        self.record_scope(self.scope_node,(body, body_index), annotate_node)
        self.log_info("[Failed]can not convert Tensor Method {} to Paddle ".format(torch_api), self.file_name, node.lineno)
        return node 


    def get_api_mather(self, torch_api):
        if torch_api in API_MAPPING:
            api_mapping = API_MAPPING[torch_api]
            if "Matcher" in api_mapping:
                matcher = api_mapping['Matcher']
                return eval(matcher)(torch_api, api_mapping)
        return None

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

        self.log_info("Mark this file which has been converted already", self.file_name)
        mark_node = ast.parse("' This file is generated by Paddle converter, you can remove this mark'").body[0]
        self.record_scope(self.root, ('body', 0), mark_node)
        return node
