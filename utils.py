
import ast
import astor
import json
import logging
from os import path

json_file = path.dirname(__file__) + "/api_mapping.json"
with open(json_file, 'r') as file:
    API_MAPPING = json.load(file)


class BaseTransformer(ast.NodeTransformer):
    def __init__(self, root, file, imports_map):
        self.root = root
        self.file = file
        self.file_name = path.basename(file)
        self.imports_map = imports_map
        self.torch_api_count = 0
        self.success_api_count = 0
        self.root = root
        self.log_msg = []
        self.node_stack = []
        self.scope_stack = []
    
    def transform(self):
        self.visit(self.root)
        
    def log_info(self, msg, file=None, line=None):
        if file:
            if line:
                msg = "[{}:{}] {}".format(file, line, msg)
            else:
                msg = "[{}] {}".format(file, msg)
        else:
            msg = "{}".format(msg)
        logging.info(msg)
        self.log_msg.append(msg)

    def get_full_attr(self, node):
        attr_list = []
        if isinstance(node, ast.Attribute):
            attr_list.insert(0, node.attr)
            self.get_full_attr(node.value)
        elif isinstance(node, ast.Name):
            attr_list.insert(0, node.id)
        else:
            attr_list.insert(0, 'TensorMethod')
        
        return '.'.join(attr_list)

    def get_full_api_from_node(self, node):
        full_attr = self.get_full_attr(node)
        attr_list = full_attr.split('.')
        old_module = attr_list[0]
        if old_module in self.imports_map[self.file]:
            new_module = self.imports_map[self.file][old_module]
            attr_list[0] = new_module
            torch_api = '.'.join(attr_list)
            return torch_api
        else:
            return None


class BaseMatcher(object):
    def __init__(self, api_mapping):
        self.api_mapping = api_mapping

    def get_paddle_api(self):
        if 'paddle_api' in self.api_mapping:
            return self.api_mapping['paddle_api']
        return None

    def args_to_kwargs(self, args, kwargs):
        args_list = self.api_mapping.get['args_list'] or []
        assert len(args) <= len(args_list)

        new_kwargs = {}
        for i, node in enumerate(args):
            k = args_list[i]
            v = astor.to_source(node).strip('\n')
            new_kwargs[k] = v
            
        for node in kwargs:
            k = node.arg
            v = astor.to_source(node.value).strip('\n')
            new_kwargs[k] = v

        return new_kwargs
    

    def get_paddle_nodes(self, args, kwargs):
        new_kwargs = self.args_to_kwargs(args, kwargs)
        new_code = self.generate_code(new_kwargs)
        if new_code:
            return ast.parse(new_code).body
        return None

    def generate_code(self, kwargs):
        raise ""
