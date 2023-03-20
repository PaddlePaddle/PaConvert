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
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            code = API_TEMPLATE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v, out_v)
        elif requires_grad_v and 'out' not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLATE.format(out, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out, out)
        elif not requires_grad_v and 'out' in kwargs:
            out_v = new_kwargs.pop('out')
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLATE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v)
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
            bias_v = eval(kwargs.pop('bias'))
            if not bias_v:
                kwargs['bias_attr'] = 'False'
        code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code
        
class TorchAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'alpha' in kwargs:
            code = "paddle.add(x={}, y={}*{})".format(kwargs['input'], kwargs['alpha'], kwargs['other'])
        else:
            code = "paddle.add(x={}, y={})".format(kwargs['input'], kwargs['other'])
        
        if 'out' in kwargs:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}
                {}
                '''
            )
            code = API_TEMPLATE.format(kwargs['out'], code, kwargs['out'])
        
        return code


class TensorAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'alpha' in kwargs:
            API_TEMPLATE = textwrap.dedent(
                '''
                paddle.Tensor.add(y={}*{})
                '''
            )
            code = API_TEMPLATE.format(kwargs['alpha'], kwargs['other'])
        else:
            API_TEMPLATE = textwrap.dedent(
                '''
                paddle.Tensor.add(y={})
                '''
            )
            code = API_TEMPLATE.format(kwargs['other'])
        return code


class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) != 3:
            return None

        API_TEMPLATE = textwrap.dedent(
            '''
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose(x=x, perm={})
            '''
        )
        perm = get_unique_name('perm')
        code = API_TEMPLATE.format(kwargs['input'], perm, 
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
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            code = API_TEMPLATE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v, out_v)
        elif requires_grad_v and 'out' not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = False
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLATE.format(out, self.get_paddle_api(), self.kwargs_to_str(kwargs), out, out)
        elif not requires_grad_v and 'out' in kwargs:
            out_v = kwargs.pop('out')
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}
                '''
            )
            code = API_TEMPLATE.format(out_v, self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v)
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

        API_TEMPLATE = textwrap.dedent(
            '''
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            x.transpose(perm={})
            '''
        )
        perm = get_unique_name('perm')
        code = API_TEMPLATE.format(self.paddleClass, perm, 
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

class TensorBF16Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='bfloat16')".format(self.paddleClass)
        return code

class TensorBoolMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='bool')".format(self.paddleClass)
        return code

class TensorByteMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='uint8')".format(self.paddleClass)
        return code

class TensorCharMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='int8')".format(self.paddleClass)
        return code

class TensorDoubleMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='float64')".format(self.paddleClass)
        return code

class TensorFloatMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='float32')".format(self.paddleClass)
        return code

class TensorFP16Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='float16')".format(self.paddleClass)
        return code

class TensorIntMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='int32')".format(self.paddleClass)
        return code

class TensorLongMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='int64')".format(self.paddleClass)
        return code

class TensorShortMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='int16')".format(self.paddleClass)
        return code

class TensorCfloatMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='complex64')".format(self.paddleClass)
        return code

class TensorCdoubleMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype='complex128')".format(self.paddleClass)
        return code

class TensorTypeAsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.astype(dtype={}.dtype)".format(self.paddleClass, kwargs['tensor'])
        return code


class CrossEntropyLossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'label_smoothing' in kwargs:
            return None

        if 'size_average' in kwargs:
            size_average = kwargs.pop('size_average')
            if 'True' in size_average:
                size_average = True
            elif 'False' in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None
        
        if 'reduce' in kwargs:
            reduce = kwargs.pop('reduce')
            if 'True' in reduce:
                reduce = True
            elif 'False' in reduce:
                reduce = False
            else:
                reduce = None
        else:
            reduce = None
        
        if size_average is not None or reduce is not None:
            if size_average is None:
                size_average = True
            if reduce is None:
                reduce = True

            if size_average and reduce:
                reduction = '"""mean"""'
            elif reduce:
                reduction = '"""sum"""'
            else:
                reduction = '"""none"""'
        elif 'reduction' in kwargs:
            reduction = kwargs.pop('reduction')
        else:
            reduction = '"""mean"""'

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.CrossEntropyLoss(weight={},
                ignore_index={},
                reduction={},
                soft_label=False,
                axis=1,
                use_softmax=True,
                name=None
            )
            '''
        )
        code = API_TEMPLACE.format(kwargs['weight'], kwargs['ignore_index'], reduction)
        return code


class LayerNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'eps' not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs['eps']

        if 'elementwise_affine' in kwargs and 'False' in kwargs['elementwise_affine']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.LayerNorm(normalized_shape={}, 
                                    epsilon={}, 
                                    weight_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    bias_attr=paddle.ParamAttr(learning_rate=0.0))
                '''
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.LayerNorm(normalized_shape={}, 
                                    epsilon={}, 
                                    weight_attr=None, 
                                    bias_attr=None)
                '''
            )
        code = API_TEMPLACE.format(kwargs['normalized_shape'], epsilon)
        return code


class GroupNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'eps' not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs['eps']

        if 'affine' in kwargs and 'False' in kwargs['affine']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.GroupNorm(num_groups={}, 
                                    num_channels={}, 
                                    epsilon={}, 
                                    weight_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    bias_attr=paddle.ParamAttr(learning_rate=0.0))
                '''
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.GroupNorm(num_groups={}, 
                                    num_channels={}, 
                                    epsilon={}, 
                                    weight_attr=None, 
                                    bias_attr=None)
                '''
            )
        code = API_TEMPLACE.format(kwargs['num_groups'], kwargs['num_channels'], epsilon)
        return code


class BatchNorm1DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'eps' not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs['eps']

        if 'track_running_stats' in kwargs:
            track_running_stats = kwargs['track_running_stats']
        else:
            track_running_stats = True

        if 'momentum' in kwargs:
            momentum = kwargs['momentum']
        else:
            momentum = 0.1

        if 'affine' in kwargs and 'False' in kwargs['affine']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.BatchNorm1D(num_features={}, 
                                    momentum=1-{}, 
                                    epsilon={}, 
                                    weight_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    bias_attr=paddle.ParamAttr(learning_rate=0.0),
                                    use_global_stats={})
                '''
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.BatchNorm1D(num_features={}, 
                                    momentum=1-{}, 
                                    epsilon={}, 
                                    weight_attr=None, 
                                    bias_attr=None,
                                    use_global_stats={})
                '''
            )
        code = API_TEMPLACE.format(kwargs['num_features'], momentum, epsilon, track_running_stats)
        return code


class BatchNorm2DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'eps' not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs['eps']

        if 'track_running_stats' in kwargs:
            track_running_stats = kwargs['track_running_stats']
        else:
            track_running_stats = True

        if 'momentum' in kwargs:
            momentum = kwargs['momentum']
        else:
            momentum = 0.1

        if 'affine' in kwargs and 'False' in kwargs['affine']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.BatchNorm2D(num_features={}, 
                                    momentum=1-{}, 
                                    epsilon={}, 
                                    weight_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    bias_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    use_global_stats={})
                '''
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.BatchNorm2D(num_features={}, 
                                    momentum=1-{}, 
                                    epsilon={}, 
                                    weight_attr=None, 
                                    bias_attr=None, 
                                    use_global_stats={})
                '''
            )
        code = API_TEMPLACE.format(kwargs['num_features'], momentum, epsilon, track_running_stats)
        return code


class BatchNorm3DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'eps' not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs['eps']

        if 'track_running_stats' in kwargs:
            track_running_stats = kwargs['track_running_stats']
        else:
            track_running_stats = True

        if 'momentum' in kwargs:
            momentum = kwargs['momentum']
        else:
            momentum = 0.1

        if 'affine' in kwargs and 'False' in kwargs['affine']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.BatchNorm3D(num_features={}, 
                                    momentum=1-{}, 
                                    epsilon={}, 
                                    weight_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    bias_attr=paddle.ParamAttr(learning_rate=0.0), 
                                    use_global_stats={})
                '''
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.BatchNorm3D(num_features={}, 
                                    momentum=1-{}, 
                                    epsilon={}, 
                                    weight_attr=None, 
                                    bias_attr=None, 
                                    use_global_stats={})
                '''
            )
        code = API_TEMPLACE.format(kwargs['num_features'], momentum, epsilon, track_running_stats)
        return code


class MaxPool2DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'stride' not in kwargs:
            stride = None
        else:
            stride = kwargs['stride']

        if 'padding' in kwargs:
            padding = kwargs['padding']
        else:
            padding = 0

        if 'return_indices' in kwargs:
            return_mask = kwargs['return_indices']
        else:
            return_mask = False

        if 'ceil_mode' in kwargs:
            ceil_mode = kwargs['ceil_mode']
        else:
            ceil_mode = False

        if 'dilation' in kwargs and kwargs['dilation'] != '(1)':
            return None

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.MaxPool2D(kernel_size={}, 
                                stride={}, 
                                padding={}, 
                                ceil_mode={}, 
                                return_mask={})
            '''
        )
        code = API_TEMPLACE.format(kwargs['kernel_size'], stride, padding, ceil_mode, return_mask)
        return code


class DivMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'out' in kwargs and kwargs['out'] != 'None':
            out = kwargs['out']
        else:
            out = None
        
        if 'rounding_mode' in kwargs and kwargs['rounding_mode'] != 'None':
            rounding_mode = kwargs['rounding_mode']
        else:
            rounding_mode = None

        if out is not None:
            if rounding_mode is not None and 'trunc' in rounding_mode:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    {} = paddle.trunc(paddle.divide(x={}, y={}))
                    {}
                    '''
                )
            elif rounding_mode is not None and 'floor' in rounding_mode:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    {} = paddle.floor(paddle.divide(x={}, y={}))
                    {}
                    '''
                )
            else:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    {} = paddle.divide(x={}, y={})
                    {}
                    '''
                )
            code = API_TEMPLACE.format(out, kwargs['input'], kwargs['other'], out)
        else:
            if rounding_mode is not None and 'trunc' in rounding_mode:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    paddle.trunc(paddle.divide(x={}, y={}))
                    '''
                )
            elif rounding_mode is not None and 'floor' in rounding_mode:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    paddle.floor(paddle.divide(x={}, y={}))
                    '''
                )
            else:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    paddle.divide(x={}, y={})
                    '''
                )
            code = API_TEMPLACE.format(kwargs['input'], kwargs['other'])
        return code


class SplitMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'dim' in kwargs:
            axis = kwargs['dim']
        else:
            axis = 0

        if '[' in kwargs['split_size_or_sections']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.split(x={}, num_or_sections={}, axis={})
                '''
            )
            code = API_TEMPLACE.format(kwargs['tensor'], kwargs['split_size_or_sections'], axis)
        else: 
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.split(x={}, num_or_sections={}.shape[{}]//{}, axis={})
                '''
            )
            code = API_TEMPLACE.format(kwargs['tensor'], kwargs['tensor'], axis, kwargs['split_size_or_sections'], axis)
        return code


class RangeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        else:
            dtype = '"""float32"""'

        if 'requires_grad' in kwargs and 'True' in kwargs['requires_grad']:
            stop_gradient = False
        else:
            stop_gradient = True
        
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0

        if 'step' in kwargs:
            step = kwargs['step']
        else:
            step = 1

        out = get_unique_name('out')
        API_TEMPLACE = textwrap.dedent(
            '''
            {} = paddle.arange(start={}, end={}+1 if ({} - {}) % {} == 0 else {}, step={}, dtype={})
            {}.stop_gradient = {}
            {}
            '''
        )
        code = API_TEMPLACE.format(out, start, kwargs['end'], kwargs['end'], start, step, kwargs['end'], step, dtype, out, stop_gradient, out)
        return code


class MeshgridMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        if 'indexing' in new_kwargs:
            if 'ij' not in new_kwargs['indexing']:
                return None
        code = '{}({})'.format(self.get_paddle_api(), self.args_to_str(new_args))
        return ast.parse(code).body


class TensorIsContiguousMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        code = 'True'
        return ast.parse(code).body
    

class TensorSkipMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        code = '{}'.format(self.paddleClass)
        return ast.parse(code).body


class TensorCopyMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        args = self.parse_args(args)
        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.assign({}, output={})
            '''
        )
        code = API_TEMPLACE.format(args[0], self.paddleClass)
        return ast.parse(code).body


class TensorMaskedFillMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            return None

        if 'value' in kwargs:
            value = kwargs['value']
        else:
            return None
        
        API_TEMPLACE = textwrap.dedent(
            '''
            detach_x = {}.detach()
            detach_x = paddle.full(detach_x.shape, {}, detach_x.dtype)
            {} = paddle.where({}, detach_x, {})
            {}
            '''
        )
        out = get_unique_name('out')
        code = API_TEMPLACE.format(self.paddleClass, value, self.paddleClass, mask, self.paddleClass, self.paddleClass)
        return ast.parse(code).body


class TensorUniqueMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if 'sorted' in kwargs and 'False' in kwargs['sorted']:
            return None

        kwargs_change = {}
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
        new_kwargs = {}
        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)

        API_TEMPLACE = textwrap.dedent(
            '''
            {}.unique({})
            '''
        )
        code = API_TEMPLACE.format(self.paddleClass, self.kwargs_to_str(new_kwargs))
        return ast.parse(code).body


class TensorExpandMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)

        if len(args) == 1 and not isinstance(args[0], ast.Constant):
            shape_list = self.parse_args(args)[0]
        else:
            shape_list = self.parse_args(args)

        kwargs = self.parse_kwargs(kwargs)
        if 'sizes' in kwargs:
            kwargs = { 'shape' : kwargs.pop('sizes'), **kwargs}
        else:
            kwargs = { 'shape' : str(shape_list).replace('\'', ''), **kwargs}

        code = '{}.expand({})'.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorSoftmaxMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if 'dim' in kwargs:
            axis = kwargs['dim']
        else:
            return None

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.softmax({}, axis={})
            '''
        )
        code = API_TEMPLACE.format(self.paddleClass, axis)
        return ast.parse(code).body


class TensorRequiresGradMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if 'requires_grad' in kwargs:
            if 'True' in kwargs['requires_grad']:
                stop_gradient = 'False'
            elif 'False' in kwargs['requires_grad']:
                stop_gradient = 'True'
            else:
                stop_gradient = kwargs['requires_grad']
        else:
            stop_gradient = 'False'

        API_TEMPLACE = textwrap.dedent(
            '''
            {}.stop_gradient = {}
            {}
            '''
        )
        code = API_TEMPLACE.format(self.paddleClass, stop_gradient, self.paddleClass)
        return ast.parse(code).body


class FunctionalL1LossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'size_average' in kwargs:
            size_average = kwargs.pop('size_average')
            if 'True' in size_average:
                size_average = True
            elif 'False' in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None
        
        if 'reduce' in kwargs:
            reduce = kwargs.pop('reduce')
            if 'True' in reduce:
                reduce = True
            elif 'False' in reduce:
                reduce = False
            else:
                reduce = None
        else:
            reduce = None
        
        if size_average is not None or reduce is not None:
            if size_average is None:
                size_average = True
            if reduce is None:
                reduce = True

            if size_average and reduce:
                reduction = '"""mean"""'
            elif reduce:
                reduction = '"""sum"""'
            else:
                reduction = '"""none"""'
        elif 'reduction' in kwargs:
            reduction = kwargs.pop('reduction')
        else:
            reduction = '"""mean"""'

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.l1_loss(input={}, label={}, reduction={})
            '''
        )
        code = API_TEMPLACE.format(kwargs['input'], kwargs['target'], reduction)
        return code


class FunctionalBinaryCrossEntropyWithLogitsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'size_average' in kwargs:
            size_average = kwargs.pop('size_average')
            if 'True' in size_average:
                size_average = True
            elif 'False' in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None
        
        if 'reduce' in kwargs:
            reduce = kwargs.pop('reduce')
            if 'True' in reduce:
                reduce = True
            elif 'False' in reduce:
                reduce = False
            else:
                reduce = None
        else:
            reduce = None
        
        if size_average is not None or reduce is not None:
            if size_average is None:
                size_average = True
            if reduce is None:
                reduce = True

            if size_average and reduce:
                reduction = '"""mean"""'
            elif reduce:
                reduction = '"""sum"""'
            else:
                reduction = '"""none"""'
        elif 'reduction' in kwargs:
            reduction = kwargs.pop('reduction')
        else:
            reduction = '"""mean"""'

        if 'weight' in kwargs:
            weight = kwargs.pop('weight')
        else:
            weight = 'None'

        if 'pos_weight' in kwargs:
            pos_weight = kwargs.pop('pos_weight')
        else:
            pos_weight = 'None'

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.binary_cross_entropy_with_logits(logit={}, label={}, weight={}, reduction={}, pos_weight={})
            '''
        )
        code = API_TEMPLACE.format(kwargs['input'], kwargs['target'], weight, reduction, pos_weight)
        return code


class FunctionalMaxPool2DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'dilation' in kwargs and kwargs['dilation'] != '(1)':
            return None
        
        if 'stride' in kwargs:
            stride = kwargs.pop('stride')
        else:
            stride = 'None'
        
        if 'padding' in kwargs:
            padding = kwargs.pop('padding')
        else:
            padding = 0
        
        if 'ceil_mode' in kwargs:
            ceil_mode = kwargs.pop('ceil_mode')
        else:
            ceil_mode = 'False'

        if 'return_indices' in kwargs:
            return_mask = kwargs.pop('return_indices')
        else:
            return_mask = 'False'

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.max_pool2d(x={}, 
                                            kernel_size={}, 
                                            stride={}, 
                                            padding={}, 
                                            ceil_mode={}, 
                                            return_mask={})
            '''
        )
        code = API_TEMPLACE.format(kwargs['input'], kwargs['kernel_size'], stride, padding, ceil_mode, return_mask)
        return code


class LoadMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        unsupported_params = ["map_location", "pickle_module", "weights_only", "pickle_load_args"]
        for param in unsupported_params:
            if param in kwargs:
                return None

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.load(path={})
            '''
        )
        code = API_TEMPLACE.format(kwargs['f'])
        return code


class SaveMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'pickle_module' in kwargs or '_use_new_zipfile_serialization' in kwargs:
            return None
        
        if 'pickle_protocol' in kwargs:
            protocol = kwargs['pickle_protocol']
        else:
            protocol = 4

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.save(obj={}, path={}, protocol={})
            '''
        )
        code = API_TEMPLACE.format(kwargs['obj'], kwargs['f'], protocol)
        return code