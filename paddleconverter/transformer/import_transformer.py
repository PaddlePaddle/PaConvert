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
from os import path

import sys
sys.path.append(path.dirname(__file__)+"../")

from paddleconverter.base import BaseTransformer

class ImportTransformer(BaseTransformer):
    '''
    Record import information
    '''

    def __init__(self, root, file, imports_map, logger):
        super(ImportTransformer, self).__init__(root, file, imports_map, logger)
        self.imports_map[self.file]['others'] = []
        self.import_paddle = False

    def visit_Import(self, node):
        '''
        1. remove import torch.nn
        2. remove import torch.nn as nn
        3. add import paddle
        '''
        new_node_names = []
        for alias_node in node.names:
            if 'torch.' in alias_node.name or 'torch' == alias_node.name:
                # If there is any torch.* module, use paddle to replace it
                self.import_paddle = True
                if alias_node.asname:
                    self.log_info("remove 'import {} as {}' ".format(alias_node.name, alias_node.asname), self.file_name, node.lineno)
                    self.imports_map[self.file][alias_node.asname] = alias_node.name
                else:
                    self.log_info("remove 'import {}' ".format(alias_node.name), self.file_name, node.lineno)
                    self.imports_map[self.file]['torch'] = 'torch'
            else:
                if alias_node.asname:
                    self.imports_map[self.file]['others'].append(alias_node.asname)
                else:
                    self.imports_map[self.file]['others'].append(alias_node.name)
                new_node_names.append(alias_node)

        if len(new_node_names) > 0:
            node.names = new_node_names
            return node
        else:
            return None
            
    def visit_ImportFrom(self, node):
        '''
        1. remove from torch import nn
        2. remove from torch import nn.functional as F
        '''
        # from . import Net
        if node.module:
            if 'torch.' in node.module or 'torch' == node.module:
                self.import_paddle = True
                for alias_node in node.names:
                    if alias_node.asname:
                        self.log_info("remove 'from {} import {} as {}' ".format(node.module, alias_node.name, alias_node.asname), self.file_name, node.lineno)
                        self.imports_map[self.file][alias_node.asname] = '.'.join([node.module, alias_node.name])
                    else:
                        self.log_info("remove 'from {} import {}' ".format(node.module, alias_node.name), self.file_name, node.lineno)
                        self.imports_map[self.file][alias_node.name] = '.'.join([node.module, alias_node.name])
                return None

        for alias_node in node.names:
            if alias_node.asname:
                self.imports_map[self.file]['others'].append(alias_node.asname)
            else:
                # from data_loader.modules import *
                if alias_node.name != '*':
                    self.imports_map[self.file]['others'].append(alias_node.name)

        return node


    def visit_Attribute(self, node):
        '''
        change torch api to full api according to import info.
        eg.
            nn.Module -> torch.nn.Module
        '''
        super(ImportTransformer, self).generic_visit(node)
        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            return ast.parse(torch_api).body[0].value
        return node

    def visit_Name(self, node):
        '''
        change torch api to full api according to import info.
        eg.
            Module -> torch.nn.Module
        '''
        super(ImportTransformer, self).generic_visit(node)
        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            return ast.parse(torch_api).body[0].value
        return node
    
    def visit_Module(self, node):
        super(ImportTransformer, self).generic_visit(node)

        if self.import_paddle:
            self.log_info("add 'import paddle' in first line", self.file_name)
            self.record_scope( (self.root, 'body', 0), ast.parse('import paddle').body)
