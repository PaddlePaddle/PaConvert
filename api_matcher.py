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

from .utils import API_MAPPING, BaseMatcher


class GenericMatcher(BaseMatcher):

    def get_paddle_api(self):
        assert 'paddle_api' in self.api_mapping
        return self.api_mapping['paddle_api']

    def generate_code(self, kwargs):
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
            for k, v in kwargs_change:
                if v == '':
                    del kwargs[k]
                elif k in ['layout', 'device', 'memory_format', 'inplace', 'generator']:
                    del kwargs[k]
                else:
                    kwargs[v] = kwargs.pop(k)
        if kwargs.has('out'):
            out_v = kwargs.pop('out')
            # will replace ast.Call with ast.Name
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api, kwargs, out_v)
            return None
        if kwargs.has('pin_memory') and kwargs['pin_memory']:
            code = '{}({}).pin_memory()'.format(self.get_paddle_api, kwargs)
        if kwargs.has('dtype'):
            dtype_v = kwargs.pop('dtype')
            code = '{}({}).astype({})'.format(self.get_paddle_api, kwargs, dtype_v)
        if kwargs.has('requires_grad') and kwargs['requires_grad']:
            # will replace ast.Call with ast.Name
            API_TEMPLACE = textwrap.dedent(
                '''
                z = {}({})
                z.stop_gradient = False
                z
                '''
            )
            code = API_TEMPLACE.format(self.get_paddle_api, kwargs)
        return code


class LayerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if kwargs.has('device'):
            del kwargs['device']
        if kwargs.has('dtype'):
            del kwargs['dtype']
        if kwargs.has('bias'):
            kwargs['bias_attr'] = kwargs.pop('bias')
        code = '{}({})'.format(self.get_paddle_api, kwargs)
        return code
        

class TensorAddMather(BaseMatcher):
    def generate_code(self, kwargs):
        if kwargs.has('alpha'):
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.Tensor.add(y={})
                '''
            )
            code = API_TEMPLACE.format(kwargs['other'])
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.Tensor.add(y={}*{})
                '''
            )
            code = API_TEMPLACE.format(kwargs['other'], kwargs['alpha'])
        return code

class ToTensorMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if kwargs.has('device'):
            kwargs['place'] = kwargs.pop('device')

        if kwargs.has('requires_grad'):
            requires_grad_v = kwargs.pop('requires_grad')
            code = 'paddle.to_tensor({}, stop_gradient={})'.format(kwargs, not requires_grad_v)
        if kwargs.has('pin_memory') and kwargs['pin_memory']:
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
        code = API_TEMPLACE.format(kwargs['input'], kwargs['dim0'], kwargs['dim1'], kwargs['dim1'], kwargs['dim0'])
        return code


class CreateMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        shape = []
        for node in args:
            shape.append(node.value)
        kwargs['shape'] = shape

        if kwargs.has('layout'):
            del kwargs['layout']
        if kwargs.has('device'):
            del kwargs['device']
        if kwargs.has('pin_memory'):
            del kwargs['pin_memory']
        if kwargs.has('dtype'):
            del kwargs['dtype']

        requires_grad = kwargs.has('requires_grad') and kwargs['requires_grad']
        if requires_grad and kwargs.has('out'):
            out_v = kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api, kwargs, out_v, out_v)
        elif requires_grad and not kwargs.has('out'):
            API_TEMPLACE = textwrap.dedent(
                '''
                z = {}({})
                z.stop_gradient = False
                z
                '''
            )
            code = API_TEMPLACE.format(self.get_paddle_api, kwargs)
        elif not requires_grad and kwargs.has('out'):
            out_v = kwargs.pop('out')
            API_TEMPLACE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLACE.format(out_v, self.get_paddle_api, kwargs, out_v)

        if kwargs.has('pin_memory') and kwargs['pin_memory']:
            code += ".pin_memory()"

        if kwargs.has('dtype'):
            code += ".astype({})".format(kwargs['dtype'])

        return ast.parse(code).body
