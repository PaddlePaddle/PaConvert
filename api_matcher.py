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

from base import API_MAPPING, BaseMatcher

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
        if 'dtype' in new_kwargs:
            dtype_v = kwargs.pop('dtype')
            code = '{}({}).astype({})'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs), dtype_v)
        if 'requires_grad' in new_kwargs and new_kwargs['requires_grad']:
            # will replace ast.Call with ast.Name
            API_TEMPLACE = textwrap.dedent(
                '''
                z = {}({})
                z.stop_gradient = False
                z
                '''
            )
            code = API_TEMPLACE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
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
            perm = range(len({}.shape))
            perm[{}] = {}
            perm[{}] = {}
            paddle.transpose({}, perm)
            '''
        )
        code = API_TEMPLACE.format(kwargs['input'], kwargs['dim0'], kwargs['dim1'], kwargs['dim1'], kwargs['dim0'], kwargs['input'])
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
                z = {}({})
                z.stop_gradient = False
                z
                '''
            )
            code = API_TEMPLACE.format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))
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
    