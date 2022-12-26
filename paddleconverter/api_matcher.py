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

from paddleconverter.base import API_MAPPING, BaseMatcher
from paddleconverter.utils import get_unique_name

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
            if k in ['layout', 'device', 'memory_format', 'inplace', 'generator']:
                continue
            if k in kwargs_change:
                if kwargs_change[k]:
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)
            else:
                #TODO: kwargs_change -> kwargs_mapping
                # not mapping in kwargs in there is not in kwargs_mapping
                new_kwargs[k] = kwargs[k]

        pin_memory_v = False
        if 'pin_memory' in kwargs:
            pin_memory_v = eval(new_kwargs.pop('pin_memory'))

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
            code = API_TEMPLACE.format(temp, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out, out)
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

        return code

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

class ToTensorMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'device' in kwargs:
            kwargs['place'] = kwargs.pop('device')

        if 'requires_grad' in kwargs:
            requires_grad_v = kwargs.pop('requires_grad')
            code = 'paddle.to_tensor({}, stop_gradient = {})'.format(self.kwargs_to_str(kwargs), not requires_grad_v)
        else:
            code = 'paddle.to_tensor({})'.format(self.kwargs_to_str(kwargs))

        if 'pin_memory' in kwargs:
            pin_memory_v = eval(kwargs['pin_memory'])
            if pin_memory_v:
                code = code + '.pin_memory()'

        return code

class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) != 3:
            return None

        API_TEMPLACE = textwrap.dedent(
            '''
            {} = list(range(len({}.shape)))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose({}, {})
            '''
        )
        perm = get_unique_name('perm')
        code = API_TEMPLACE.format(perm, kwargs['input'], 
                perm, kwargs['dim0'], kwargs['dim1'], 
                perm, kwargs['dim1'], kwargs['dim0'], 
                kwargs['input'], perm)
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
        if len(args) > 1:
            return None
        device_str = args[0].value

        code = "'{}'".format(device_str)
        return ast.parse(code).body
    

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
        if len(args) == 1 and isinstance(args[0], ast.Call):
            if self.get_full_attr(args[0].func).endswith('OrderedDict'):
                new_args = self.parse_args(args[0].args)
                new_args = ['*{}'.format(new_args[0])]
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
        call_maximum = False
        if len(args) > 1 and isinstance(args[1], ast.Name):
            call_maximum = True
        
        kwargs = self.parse_kwargs(kwargs)
        if 'other' in kwargs:
            call_maximum = True
        
        if call_maximum:
            return GenericMatcher(self.torch_api, self.api_mapping).get_paddle_nodes(args, kwargs)

        # return (values, indices) and paddle not implement
        if len(args) > 1 and isinstance(args[1], ast.Num):  
            return None
        if 'dim' in kwargs:
            return None

        # only return values
        if 'input' in kwargs:
            x_v = astor.to_source(kwargs['input']).strip('\n')
        else:
            x_v = astor.to_source(args[0]).strip('\n')

        code = 'paddle.max({})'.format(x_v)
        return ast.parse(code).body

class TensorMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        return None


class TensorTransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        # may be ndarray.transpose([list]) / ndarray.transpose(list)
        if len(kwargs) != 2:
            return "NonTorchTensor"

        API_TEMPLACE = textwrap.dedent(
            '''
            {} = list(range(len({}.shape)))
            {}[{}] = {}
            {}[{}] = {}
            {}.transpose({})
            '''
        )
        perm = get_unique_name('perm')
        code = API_TEMPLACE.format(perm, self.paddleTensor, 
                perm, kwargs['dim0'], kwargs['dim1'], 
                perm, kwargs['dim1'], kwargs['dim0'], 
                self.paddleTensor, perm)
        return code


class TensorReshapeMatcher(BaseMatcher):
    def get_paddle_tensor_nodes(self, func, args, kwargs):
        self.parse_func(func)
        if 'shape' in self.parse_kwargs(kwargs):
            kwargs = self.parse_kwargs(kwargs)    
        elif len(args) == 1 and isinstance(args[0], ast.List):
            kwargs = self.parse_args_and_kwargs(args, kwargs)
        else:
            shape_list = self.parse_args(args)
            kwargs = {'shape': str(shape_list).replace('\'', '')}

        code = '{}.reshape({})'.format(self.paddleTensor, self.kwargs_to_str(kwargs))
        return ast.parse(code).body

class TensorSizeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'dim' in kwargs:
            code = '{}.shape[{}]'.format(self.paddleTensor, kwargs['dim'])
        else:
            code = '{}.shape'.format(self.paddleTensor)
        return code
    
class TensorRequiresGradMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'requires_grad' in kwargs:
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}
                {}.stop_gradient = not {}
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLACE.format(out, self.paddleTensor, out, kwargs.pop('requires_grad'), self.paddleTensor)
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}
                {}.stop_gradient = False
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLACE.format(out, self.paddleTensor, out, out)

        return code

class TensorPermuteMatcher(BaseMatcher):
    def get_paddle_tensor_nodes(self, func, args, kwargs):
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

        code = '{}.transpose({})'.format(self.paddleTensor, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorViewMatcher(BaseMatcher):
    def get_paddle_tensor_nodes(self, func, args, kwargs):
        self.parse_func(func)

        kwargs = self.parse_kwargs(kwargs)
        if 'dtype' in kwargs:
            if 'np' in kwargs['dtype'] or 'numpy' in kwargs['dtype']:
                return 'NonTensor'
            else:
                return None

        if len(args) == 1:
            if isinstance(args[0], ast.Attribute):
                return 'NonTensor'
            if isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
                return None
            
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            shape_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            shape_list = self.parse_args(args)

        if 'size' in kwargs:
            kwargs = { 'shape' : kwargs.pop('size'), **kwargs}
        else:
            kwargs = { 'shape' : str(shape_list).replace('\'', ''), **kwargs}

        code = '{}.reshape({})'.format(self.paddleTensor, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorRepeatMatcher(BaseMatcher):
    def get_paddle_tensor_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)

        if 'axis' in kwargs:
            return 'NonTensor'

        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            repeat_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            repeat_list = self.parse_args(args)

        if 'repeats' in kwargs:
            kwargs = { 'repeat_times' : kwargs.pop('repeats'), **kwargs}
        else:
            kwargs = { 'repeat_times' : str(repeat_list).replace('\'', ''), **kwargs}
            
        code = '{}.tile({})'.format(self.paddleTensor, self.kwargs_to_str(kwargs))
        return ast.parse(code).body

