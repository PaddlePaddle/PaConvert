import ast
import logging
from os import path

import sys
sys.path.append(path.dirname(__file__)+"../")

from base import BaseTransformer

class ImportTransformer(BaseTransformer):
    '''
    Record import information
    '''

    def __init__(self, root, file, imports_map):
        super(ImportTransformer, self).__init__(root, file, imports_map)
        self.imports_map[self.file]['others'] = []

    def visit_Import(self, node):
        '''
        1. remove import torch.nn
        2. remove import torch.nn as nn
        3. add import paddle
        '''
        new_node_names = []
        for alias_node in node.names:
            if 'torch' in alias_node.name:
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
        if 'torch' in node.module:
            for alias_node in node.names:
                if alias_node.asname:
                    self.log_info("remove 'from {} import {} as {}' ".format(node.module, alias_node.name, alias_node.asname), self.file_name, node.lineno)
                    self.imports_map[self.file][alias_node.asname] = '.'.join([node.module, alias_node.name])
                else:
                    self.log_info("remove 'from {} import {}' ".format(node.module, alias_node.name), self.file_name, node.lineno)
                    self.imports_map[self.file][alias_node.name] = '.'.join([node.module, alias_node.name])
            return None
        else:
            for alias_node in node.names:
                if alias_node.as_name:
                    self.imports_map[self.file]['others'].append(alias_node.asname)
                else:
                    self.imports_map[self.file]['others'].append(alias_node.name)

        return node


    def visit_Attribute(self, node):
        '''
        change torch api to full api according to import info.
        eg.
            nn.Module -> torch.nn.Module
        '''
        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            self.torch_api_count += 1
            return ast.parse(torch_api).body[0].value
        return node

    def visit_Name(self, node):
        '''
        change torch api to full api according to import info.
        eg.
            Module -> torch.nn.Module
        '''
        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            self.torch_api_count += 1
            return ast.parse(torch_api).body[0].value
        return node
    
    def visit_Module(self, node):
        super(ImportTransformer, self).generic_visit(node)
        self.log_info("Mark this file which is converted", self.file_name)
        mark_node = ast.parse("' This file has been converted by Paddle converter, thanks to use, you can remove this mark~'").body[0]
        self.record_scope(self.root, 0, mark_node)
        
        self.log_info("Will add 'import paddle' in first line", self.file_name)
        self.record_scope(self.root, 0, ast.parse('import paddle').body)
