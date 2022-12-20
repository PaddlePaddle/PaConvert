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
from paddleconverter.utils import unique_name

class GenericMatcher(BaseMatcher):

    def get_paddle_api(self):
        assert 'paddle_api' in self.api_mapping
        return self.api_mapping['paddle_api']

    def generate_code(self, kwargs):
        kwargs_change = {}
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
        new_kwargs = {}
        for k, v in kwargs.items():
            if k in ['layout', 'device', 'memory_format', 'inplace', 'generator']:
                continue
            if k in kwargs_change:
                if v:
                    new_kwargs[kwargs_change[k]] = v
            else:
                new_kwargs[k] = v

        
        code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))
        if 'out' in new_kwargs:
            out_v = new_kwargs.pop('out')
            # will replace ast.Call with ast.Name
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v)
        if 'pin_memory' in new_kwargs and new_kwargs['pin_memory']:
            code = '{}({}).pin_memory()'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        if 'requires_grad' in new_kwargs and new_kwargs['requires_grad']:
            # will replace ast.Call with ast.Name
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            temp = unique_name('temp')
            code = API_TEMPLACE.format(temp, self.get_paddle_api(), self.kwargs_to_str(kwargs), temp, temp)
        return code


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
        if 'pin_memory' in kwargs and kwargs['pin_memory']:
            code = code + '.pin_memory()'

        return code

class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLACE = textwrap.dedent(
            '''
            {} = list(range(len({}.shape)))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose({}, {})
            '''
        )
        perm = unique_name('perm')
        code = API_TEMPLACE.format(perm, kwargs['input'], 
                perm, kwargs['dim0'], kwargs['dim1'], 
                perm, kwargs['dim1'], kwargs['dim0'], 
                kwargs['input'], perm)
        return code


class CreateMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_kwargs = {}
        shape_list = []
        for node in args:
            shape_list.append(node.value)
        new_kwargs['shape'] = str(shape_list)
        for node in kwargs:
            k = node.arg
            v = astor.to_source(node.value).strip('\n')
            new_kwargs[k] = v

        if 'layout' in kwargs:
            del new_kwargs['layout']
        if 'device' in kwargs:
            del new_kwargs['device']
        if 'pin_memory' in kwargs:
            del new_kwargs['pin_memory']
        if 'dtype' in kwargs:
            del new_kwargs['dtype']

        requires_grad = ('requires_grad' in new_kwargs) and new_kwargs['requires_grad']
        if requires_grad and 'out' in new_kwargs:
            new_kwargs.pop('requires_grad')
            out_v = new_kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v, out_v)
        elif requires_grad and 'out' not in new_kwargs:
            new_kwargs.pop('requires_grad')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            temp = unique_name('temp')
            code = API_TEMPLACE.format(temp, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), temp, temp)
        elif not requires_grad and 'out' in new_kwargs:
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

        if 'pin_memory' in new_kwargs and new_kwargs['pin_memory']:
            code += ".pin_memory()"
        
        return ast.parse(code).body


class DeviceMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args)>1:
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
                new_args = self.parse_args(args[0].args[0].elts)
        else:
            new_args = self.parse_args(args)
        code = 'paddle.nn.Squential({})'.format(self.args_to_str(new_args))
        return ast.parse(code).body

class IdentityMatcher(BaseMatcher):

    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        code = 'paddle.nn.Identity({}, {})'.format(self.args_to_str(new_args), self.kwargs_to_str(new_kwargs))
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
