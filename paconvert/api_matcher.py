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
import astor
import textwrap

from paconvert.base import API_MAPPING, BaseMatcher
from paconvert.utils import get_unique_name

class GenericMatcher(BaseMatcher):

    def get_paddle_api(self):
        assert 'paddle_api' in self.api_mapping
        if self.paddle_api:
            return self.paddle_api
        return self.api_mapping['paddle_api']

    def generate_code(self, kwargs):
        kwargs_change = {}
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
        new_kwargs = {}
        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)
            else:
                # remove directly and not handle
                if k in ['layout', 'device', 'memory_format', 'inplace', 'generator', 'non_blocking']:
                    kwargs.pop(k)
                    continue
                
                #TODO: kwargs_change -> kwargs_mapping
                # not mapping in kwargs in there is not in kwargs_mapping
                new_kwargs[k] = kwargs[k]

        pin_memory_v = False
        if 'pin_memory' in kwargs:
            pin_memory_v = eval(new_kwargs.pop('pin_memory'))

        dtype_v = None
        if 'dtype' in kwargs:
            dtype_v = new_kwargs.pop('dtype')

        requires_grad_v = False
        if 'requires_grad' in kwargs:
            requires_grad_v = eval(new_kwargs.pop('requires_grad'))

        if requires_grad_v and 'out' in kwargs:
            out_v = new_kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v, out_v)
        elif requires_grad_v and 'out' not in kwargs:
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLACE.format(out, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out, out)
        elif not requires_grad_v and 'out' in kwargs:
            out_v = new_kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v)
        else:
            code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))

        if pin_memory_v:
            code = code.rstrip('\n') + ".pin_memory()"

        if dtype_v:
            code = code.rstrip('\n') + ".astype({})".format(dtype_v)
            
        return code

class DeleteMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        return 'delete'

    def get_paddle_api(self):
        return 'delete'


class IdentityMatcher(BaseMatcher):

    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        code = '{}({})'.format(self.get_paddle_api(), self.args_and_kwargs_to_str(new_args, new_kwargs))
        return ast.parse(code).body

class LayerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'device' in kwargs:
            del kwargs['device']
        if 'dtype' in kwargs:
            del kwargs['dtype']
        if 'bias' in kwargs:
            kwargs['bias_attr'] = kwargs.pop('bias')
        code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code
        
class TorchAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'out' in kwargs:
            return None

        if 'alpha' in kwargs:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.add(x={}, y={}*{})
                '''
            )
            code = API_TEMPLACE.format(kwargs['alpha'], kwargs['input'], kwargs['other'])
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.add(x={}, y={})
                '''
            )
            code = API_TEMPLACE.format(kwargs['input'], kwargs['other'])
        return code


class TensorAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'alpha' in kwargs:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.Tensor.add(y={}*{})
                '''
            )
            code = API_TEMPLACE.format(kwargs['alpha'], kwargs['other'])
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.Tensor.add(y={})
                '''
            )
            code = API_TEMPLACE.format(kwargs['other'])
        return code


class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) != 3:
            return None

        API_TEMPLACE = textwrap.dedent(
            '''
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose(x=x, perm={})
            '''
        )
        perm = get_unique_name('perm')
        code = API_TEMPLACE.format(kwargs['input'], perm, 
                perm, kwargs['dim0'], kwargs['dim1'], 
                perm, kwargs['dim1'], kwargs['dim0'], 
                perm)
        return code


class CreateMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            shape_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            shape_list = self.parse_args(args)

        kwargs = self.parse_kwargs(kwargs)
        if 'size' in kwargs:
            kwargs = { 'shape' : kwargs.pop('size'), **kwargs}
        else:
            kwargs = { 'shape' : str(shape_list).replace('\'', ''), **kwargs}

        if 'layout' in kwargs:
            del kwargs['layout']
        if 'device' in kwargs:
            del kwargs['device']
        if 'memory_format' in kwargs:
            del kwargs['memory_format']
        
        pin_memory_v = False
        if 'pin_memory' in kwargs:
            pin_memory_v = eval(kwargs.pop('pin_memory'))
        
        requires_grad_v = False
        if 'requires_grad' in kwargs:
            requires_grad_v = eval(kwargs.pop('requires_grad'))
        
        if requires_grad_v and 'out' in kwargs:
            out_v = kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v, out_v)
        elif requires_grad_v and 'out' not in kwargs:
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLACE.format(out, self.get_paddle_api(), self.kwargs_to_str(kwargs), out, out)
        elif not requires_grad_v and 'out' in kwargs:
            out_v = kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v)
        else:
            code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        if pin_memory_v:
            code = code.rstrip('\n') + ".pin_memory()"
  
        return ast.parse(code).body


class DeviceMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args) == 1 and isinstance(args[0], ast.Str):
            device_str = args[0].value
            valid = False
            for ele in ['cpu', 'cuda', 'ipu', 'xpu']:
                if ele in device_str:
                    valid = True
            if not valid:
                return None
            
            if 'cuda' in device_str:
                device_str = device_str.replace('cuda', 'gpu')

            code = "'{}'".format(device_str)
            return ast.parse(code).body
        
        return None
    

class GeluMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'input' in kwargs:
            kwargs['x'] = kwargs.pop('input')

        if 'approximate' in kwargs:
            approximate_v = kwargs.pop('approximate')
            if 'none' in approximate_v:
                kwargs['approximate'] = 'False'
            elif 'tanh' in approximate_v:
                kwargs['approximate'] = 'True'

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code

class SquentialMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        # nn.Sequential(OrderedDict([...]) / nn.Sequential(OrderedDict(blocks))
        if len(args) == 1 and isinstance(args[0], ast.Call) and self.get_full_attr(args[0].func).endswith('OrderedDict'):
            new_args = self.parse_args(args[0].args)
            new_args = ['*{}'.format(new_args[0])]
        # nn.Sequential(module1, module2, ...)
        else:
            new_args = self.parse_args(args)
        code = 'paddle.nn.Sequential({})'.format(self.args_to_str(new_args))
        return ast.parse(code).body

class PadMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'Reflection' in self.torch_api:
            kwargs['mode'] = "'reflect'"
        elif 'Replication' in self.torch_api:
            kwargs['mode'] = "'replicate'"
        elif 'Constant' in self.torch_api:
            kwargs['mode'] = "'constant'"
        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class MaxMinMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        # call maximum usage, convert
        call_maximinimum = False
        if len(args) > 1 and isinstance(args[1], ast.Name):
            call_maximinimum = True
        
        new_kwargs = self.parse_kwargs(kwargs)
        if 'other' in new_kwargs:
            call_maximinimum = True
        
        if call_maximinimum:
            return GenericMatcher(self.torch_api, self.api_mapping).get_paddle_nodes(args, kwargs)

        # return (values, indices) and paddle not implement, not convert
        if len(args) > 1 and isinstance(args[1], ast.Num):  
            return None
        if 'dim' in new_kwargs:
            return None

        # only return values, not return indices, convert
        paddle_api = self.torch_api.replace('torch', 'paddle')
        if len(args) == 1:
            x_v = astor.to_source(args[0]).strip('\n')
            return ast.parse('{}(x={})'.format(paddle_api, x_v)).body

        if 'input' in new_kwargs:
            x_v = new_kwargs['input']
            return ast.parse('{}(x={})'.format(paddle_api, x_v)).body

        return None


class TensorMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        return None


class TensorTransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        # may be ndarray.transpose([list]) / ndarray.transpose(list)
        if len(kwargs) != 2:
            return "NonTorchClass"

        API_TEMPLACE = textwrap.dedent(
            '''
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            x.transpose(perm={})
            '''
        )
        perm = get_unique_name('perm')
        code = API_TEMPLACE.format(self.paddleClass, perm, 
                perm, kwargs['dim0'], kwargs['dim1'], 
                perm, kwargs['dim1'], kwargs['dim0'], perm)
        return code


class TensorSizeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'dim' in kwargs:
            code = '{}.shape[{}]'.format(self.paddleClass, kwargs['dim'])
        else:
            code = '{}.shape'.format(self.paddleClass)
        return code
    

class TensorPermuteMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            perm_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            perm_list = self.parse_args(args)

        kwargs = self.parse_kwargs(kwargs)
        if 'dims' in kwargs:
            kwargs = { 'perm' : kwargs.pop('dims'), **kwargs}
        else:
            kwargs = { 'perm' : str(perm_list).replace('\'', ''), **kwargs}

        code = '{}.transpose({})'.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body



class TensorRepeatMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)

        if 'axis' in kwargs:
            return 'NonTorchClass'

        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            repeat_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            repeat_list = self.parse_args(args)

        if 'repeats' in kwargs:
            kwargs = { 'repeat_times' : kwargs.pop('repeats'), **kwargs}
        else:
            kwargs = { 'repeat_times' : str(repeat_list).replace('\'', ''), **kwargs}
            
        code = '{}.tile({})'.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorIntMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='int32')".format(self.paddleClass)
        return code

class TensorLongMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='int64')".format(self.paddleClass)
        return code

class TensorFloatMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='float32')".format(self.paddleClass)
        return code

class TensorDoubleMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='float64')".format(self.paddleClass)
        return code


class TensorTypeAsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype={}.dtype)".format(self.paddleClass, kwargs['tensor'])
        return code
