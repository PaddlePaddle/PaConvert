
import json
import os
import ast

from ..api_matcher import *
from ..utils import API_MAPPING, BaseTransformer

class BasicTransformer(BaseTransformer):
    def visit(self, node):
        self.node_stack.append(node)
        node = super(BasicTransformer, self).visit(node)
        self.node_stack.pop()
        return node

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
                        return index
        
        return 0

    def visit_Attribute(self, node):
        '''
        torch api is not used by funcition call, such as class inherit
        '''
        if isinstance(self.parent_node, ast.Call):
            return

        full_attr = self.get_full_attr(node)
        if full_attr.startswith('torch'):
            torch_api = full_attr
            matcher = self.get_api_mather(torch_api)
            if matcher:
                paddle_api = matcher.get_paddle_api()
                if paddle_api:
                    self.success_api_count += 1
                    self.log_info("[Success]convert {} ---> {}".format(torch_api, paddle_api), self.file_name, node.lineno)
                    return ast.parse(paddle_api).body[0]

            self.log_info("[Failed]convert {} ---> {}".format(torch_api, paddle_api), self.file_name, node.lineno)
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
        if not isinstance(node.func, ast.Attribute):
            return
        
        # Postorder traversal                
        if isinstance(node.func.value, ast.Call):
             # Tensor method func, such as : x.abs().add
            self.visit(node.func.value)
            full_attr = self.get_full_attr(node.func)
            if full_attr.startswith('torch.Tensor'):
                self.torch_api_count += 1
                node = self.trans_tensor_method(node, full_attr)
        else:
            super(BasicTransformer, self).generic_visit(node)
            full_attr = self.get_full_attr(node.func)
            
            # Tensor method func, such as : x.add
            if full_attr.startswith('torch') and len(full_attr.split('.')) == 2:
                attr_list = full_attr.split('.')
                # Avoid ' np.add, scipy.add ... '
                if attr_list[0] not in self.imports_map[self.file]['others']:
                    attr_list[0] = 'torch.Tensor'
                    full_attr = '.'.join(attr_list)
                    self.torch_api_count += 1
                    
            if full_attr.startswith('torch'):
                if full_attr.startswith('torch.Tensor'):
                    return self.trans_tensor_method(node, full_attr)

                matcher = self.get_api_mather(full_attr)
                if matcher:
                    node_list = matcher.get_paddle_nodes(node.args, node.keywords)
                    if node_list:
                        new_node = node_list[-1]
                        # ast.Expr, which contain ast.Call or ast.Name
                        if isinstance(new_node, ast.Expr):
                            new_node = new_node.value
                        
                        if isinstance(new_node, (ast.Call, ast.Name)):
                            self.success_api_count += 1
                            self.log_info("[Success]convert {} to Paddle API ".format(torch_api), self.file_name, node.lineno)
                            
                            # if multiple line(node), insert into this api scope 
                            body_index = self.scope_body_index()
                            for i in range(len(node_list)-1):
                                if i == 0:
                                    self.log_info("insert extra {} lines for torch api {}".format(len(node_list)-1, torch_api), self.file_name, node.lineno)
                                self.scope_node.body.insert(body_index, node_list[i])

                            return new_node

                self.log_info("[Failed]convert {} to Paddle API ".format(torch_api), self.file_name, node.lineno)
        return node

    def trans_tensor_method(self, node, torch_api):
        body_index = self.scope_body_index()  
        matcher = self.get_api_mather(torch_api)
        if matcher:
            node_list = matcher.get_paddle_nodes(node.args, node.keywords)
            if node_list:
                new_node = node_list[-1]
                # ast.Expr which contain ast.Call or ast.Name
                if isinstance(new_node, ast.Expr):
                    new_node = new_node.value

                # for tensor method api, the last line must be 'ast.Call'
                if isinstance(new_node, ast.Call):
                    self.success_api_count += 1
                    self.log_info("[Success]convert Tensor Method API: {} to Paddle API ".format(torch_api), self.file_name, node.lineno)

                    # if multiple line, insert extra into this api scope 
                    body_index = self.scope_body_index()
                    for i in range(len(node_list)-1):
                        if i == 0:
                            self.log_info("insert extra {} lines for torch api {}".format(len(node_list)-1, torch_api), self.file_name, node.lineno)
                        self.scope_node.body.insert(body_index, node_list[i])

                    # for tensor method , only change .add(other=y)
                    node.func.attr = new_node.func.attr
                    node.args = new_node.args
                    node.keywords = new_node.keywords
                    return node

        annotate_node = ast.parse("Tensor Method, can't convert, should check whether need to convert manually").body[0]
        # only insert once avoid annotation too much
        if self.parent_node == self.scope_node:
            self.scope_node.body.insert(body_index, annotate_node)

        self.log_info("[Failed]convert Tensor Method API: {} to Paddle API ".format(torch_api), self.file_name, node.lineno)
        return node 


    def get_api_mather(self, torch_api):
        if torch_api in API_MAPPING:
            api_mapping = API_MAPPING[torch_api]
            matcher = api_mapping['Matcher']
            return eval(matcher)(api_mapping)
        return None       

    def visit_FunctionDef(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()
    
    def visit_While(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

    def visit_If(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

    def visit_Try(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

    def visit_TryFinally(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

    def visit_Module(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

    def visit_For(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

    def visit_With(self, node):
        self.scope_stack.append(node)
        super(BasicTransformer, self).generic_visit(node)
        self.scope_stack.pop()

