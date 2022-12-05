
import ast
import os
import logging

from ..utils import BaseTransformer


class ImportTransformer(BaseTransformer):
    '''
    record import information
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
        for son_node in node.names:
            if 'torch' in son_node.name:
                if son_node.hasattr('asname'):
                    self.log_info("remove import {} as {}".format(son_node.name, son_node.asname), self.file_name, son_node.lineno)
                    self.imports_map[self.file][son_node.asname] = son_node.name
                else:
                    self.log_info("remove import {}".format(son_node.name), self.file_name, son_node.lineno)
                    self.imports_map[self.file]['torch'] = 'torch'
            else:
                if son_node.hasattr('asname'):
                    self.imports_map[self.file]['others'].append(son_node.asname)
                else:
                    self.imports_map[self.file]['others'].append(son_node.name)
                new_node_names.append(son_node)

        self.log_info("Add import paddle in first line", self.file_name)
        self.root.body.insert(0, ast.parse('import paddle').body[0])
        
        node.names = new_node_names
        return node
            
    def visit_ImportFrom(self, node):
        '''
        1. remove from torch import nn
        2. remove from torch import nn.functional as F
        '''
        if 'torch' in node.module:
            for son_node in node.names:
                if son_node.hasattr('asname'):
                    self.log_info("remove from {} import {} as ".format(node.module, son_node.name, son_node.asname), self.file_name, son_node.lineno)
                    self.imports_map[self.file][son_node.asname] = '.'.join([node.module, son_node.name])
                else:
                    self.log_info("remove from {} import {}".format(node.module, son_node.name), self.file_name, son_node.lineno)
                    self.imports_map[self.file][son_node.name] = '.'.join([node.module, son_node.name])
            return None
        else:
            for son_node in node.names:
                if son_node.hasattr('asname'):
                    self.imports_map[self.file]['others'].append(son_node.asname)
                else:
                    self.imports_map[self.file]['others'].append(son_node.name)

        return node


    def visit_Attribute(self, node):
        '''
        change torch api call to full api according to import info
        '''
        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            self.torch_api_count += 1
            if 'torch.Tensor' not in torch_api:
                return ast.parse(torch_api).body[0]
        return node

    def visit_Name(self, node):
        torch_api = self.get_full_api_from_node(node)
        if torch_api:
            self.torch_api_count += 1
            return ast.parse(torch_api).body[0]
        return node 
    
    
