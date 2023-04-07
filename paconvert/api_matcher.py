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
        dtype_v = None
        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    # rename/copy in new_kwargs
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)
                else:
                    # remove in new_kwargs
                    kwargs.pop(k)
            else:
                # copy to new_kwargs
                new_kwargs[k] = kwargs.pop(k)
                
                # common process for some args
                if k in ['layout', 'device', 'memory_format', 'inplace', 'generator', 'non_blocking']:
                    new_kwargs.pop(k)
                    continue
                if k == 'dtype':
                    dtype_v = new_kwargs.pop('dtype')

        pin_memory_v = False
        if 'pin_memory' in new_kwargs:
            pin_memory_v = eval(new_kwargs.pop('pin_memory'))

        stop_gradient_v = None
        if 'requires_grad' in new_kwargs:
            stop_gradient_v = 'not ' + new_kwargs.pop("requires_grad").strip('()')

        new_kwargs = self.set_paddle_default_kwargs(new_kwargs)
        
        if stop_gradient_v and 'out' in new_kwargs:
            out_v = new_kwargs.pop('out')
            API_TEMPLATE = textwrap.dedent(
                '''
                x = {}({})
                x.stop_gradient = {}
                paddle.assign(x, output={})
                '''
            )
            code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs), stop_gradient_v, out_v)
        elif stop_gradient_v and 'out' not in new_kwargs:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = {}
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLATE.format(out, self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out, stop_gradient_v, out)
        elif not stop_gradient_v and 'out' in new_kwargs:
            out_v = new_kwargs.pop('out')
            API_TEMPLATE = textwrap.dedent(
                '''
                paddle.assign({}({}), output={})
                '''
            )
            code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v)
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
        if 'padding_mode' in kwargs:
            # TODO: just not support now
            return None
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
                paddle.assign({}, output={})
                '''
            )
            code = API_TEMPLATE.format(code, kwargs['out'])
        
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
        kwargs = self.parse_kwargs(kwargs)
        if 'size' in kwargs:
            kwargs = { 'shape' : kwargs.pop('size'), **kwargs}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip('\n')
            else:
                shape = self.parse_args(args)[0]
                
            kwargs = { 'shape' : str(shape).replace('\'', ''), **kwargs}

        for k in ['layout', 'device', 'memory_format']:
            if k in kwargs:
                kwargs.pop(k)
        
        pin_memory_v = False
        if 'pin_memory' in kwargs:
            pin_memory_v = eval(kwargs.pop('pin_memory'))
        
        stop_gradient_v = None
        if 'requires_grad' in kwargs:
            stop_gradient_v = 'not ' + kwargs.pop("requires_grad").strip('()')

        if stop_gradient_v and 'out' in kwargs:
            out_v = kwargs.pop('out')
            API_TEMPLATE = textwrap.dedent(
                '''
                x = {}({})
                x.stop_gradient = {}
                paddle.assign(x, output={})
                '''
            )
            code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs), stop_gradient_v, out_v)
        elif stop_gradient_v and 'out' not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = {}
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLATE.format(out, self.get_paddle_api(), self.kwargs_to_str(kwargs), out, stop_gradient_v, out)
        elif not stop_gradient_v and 'out' in kwargs:
            out_v = kwargs.pop('out')
            API_TEMPLATE = textwrap.dedent(
                '''
                paddle.assign({}({}), output={})
                '''
            )
            code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v)
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


class EqualMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = {}
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
        new_kwargs = {}

        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)
        
        API_TEMPLATE = textwrap.dedent(
            '''
            {}({}).item()
            '''
        )

        code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))
        return code.strip('\n')


class TensorMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if 'size' in kwargs:
            shape = kwargs.pop('size')
        else:
            if len(args) == 0:
                # torch has bug, treat 0D as 0-Size, but paddle not support 0-size
                return None
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip('\n')
            else:
                # TODO(hjf):may should use paddle.to_tensor
                return None
            shape = str(shape).replace('\'', '')

        if "torch.IntTensor" == self.torch_api:
            code = "paddle.empty(shape={}, dtype='int32')".format(shape)
        elif "torch.LongTensor" == self.torch_api:
            code = "paddle.empty(shape={}, dtype='int64')".format(shape)
        elif "torch.FloatTensor" == self.torch_api:
            code = "paddle.empty(shape={}, dtype='float32')".format(shape)
        else:
            code = "paddle.empty(shape={})".format(shape)

        node = ast.parse(code.strip('\n')).body
        return node


class RandintMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if ("high" in kwargs and kwargs["high"].startswith('[')):
            kwargs["shape"] = kwargs["high"]
            kwargs["high"] = kwargs["low"]
            kwargs["low"] = '0'

        code = GenericMatcher.generate_code(self, kwargs)
        
        return code


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


class TensorNew_Matcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if 'size' in kwargs:
            kwargs = { 'shape' : kwargs.pop('size'), **kwargs}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip('\n')
            else:
                shape = self.parse_args(args)[0]
                
            kwargs = { 'shape' : str(shape).replace('\'', ''), **kwargs}

        for k in ['layout', 'device', 'memory_format']:
            if k in kwargs:
                kwargs.pop(k)

        stop_gradient_v = None
        if 'requires_grad' in kwargs:
            stop_gradient_v = 'not ' + kwargs.pop("requires_grad").strip('()')

        pin_memory_v = False
        if 'pin_memory' in kwargs:
            pin_memory_v = eval(kwargs.pop('pin_memory'))

        if 'dtype' not in kwargs:
            kwargs['dtype'] = '{}.dtype'.format(self.paddleClass)

        if stop_gradient_v:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}({})
                {}.stop_gradient = {}
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLATE.format(out, self.get_paddle_api(), self.kwargs_to_str(kwargs), out, stop_gradient_v, out)
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        if pin_memory_v:
            code = code.rstrip('\n') + ".pin_memory()"

        return ast.parse(code.strip('\n')).body


class TensorNewFullMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = {"shape": kwargs.pop("size"), **kwargs}
        for k in ['layout', 'device', 'memory_format']:
            if k in kwargs:
                kwargs.pop(k)

        stop_gradient_v = None
        if 'requires_grad' in kwargs:
            stop_gradient_v = 'not ' + kwargs.pop("requires_grad").strip('()')

        pin_memory_v = False
        if 'pin_memory' in kwargs:
            pin_memory_v = eval(kwargs.pop('pin_memory'))

        if 'dtype' not in kwargs:
            kwargs['dtype'] = '{}.dtype'.format(self.paddleClass)

        if stop_gradient_v:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = paddle.full({})
                {}.stop_gradient = {}
                {}
                '''
            )
            out = get_unique_name('out')
            code = API_TEMPLATE.format(out, self.kwargs_to_str(kwargs), out, stop_gradient_v, out)
        else:
            code = "paddle.full({})".format(self.kwargs_to_str(kwargs))

        if pin_memory_v:
            code = code.rstrip('\n') + ".pin_memory()"
        
        return code.strip('\n')


class TensorNewTensorMatcher(BaseMatcher):

    def generate_code(self, kwargs):
        if "layout" in kwargs:
            kwargs.pop("layout")

        if "device" in kwargs:
            kwargs.pop("device")

        if "requires_grad" in kwargs:
            kwargs["stop_gradient"] = 'not ' + kwargs.pop("requires_grad").strip('()')

        if 'pin_memory' in kwargs:
            if eval(kwargs['pin_memory']):
                kwargs['place'] = 'paddle.CUDAPinnedPlace()'
            kwargs.pop('pin_memory')

        if 'dtype' in kwargs:
            code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        else:
            API_TEMPLATE = textwrap.dedent(
                '''
                {} = {}
                {}({}).astype(str({})[7:])
                '''
            )
            var = get_unique_name('var')
            code = API_TEMPLATE.format(var, self.paddleClass,
                                self.get_paddle_api(), self.kwargs_to_str(kwargs), var + '.dtype')

        return code.strip('\n')


class TorchTensorMatcher(BaseMatcher):

    def generate_code(self, kwargs):

        if "device" in kwargs:
            kwargs['place'] = kwargs.pop("device")

        if "requires_grad" in kwargs:
            kwargs["stop_gradient"] = 'not ' + kwargs.pop("requires_grad").strip('()')

        if 'pin_memory' in kwargs:
            if eval(kwargs['pin_memory']):
                kwargs['place'] = 'paddle.CUDAPinnedPlace()'
            kwargs.pop('pin_memory')
            
        code = '{}({})'.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code.strip('\n')

        
class TensorNormal_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs['shape'] = 'x.shape'
        API_TEMPLATE = textwrap.dedent(
            '''
            x = {}
            paddle.assign(paddle.normal({}).astype(x.dtype), x)
            '''
        )
        code = API_TEMPLATE.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return code.strip('\n')


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

            kwargs['reduction'] = reduction

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.CrossEntropyLoss({})
            '''
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class CudaIsAvailableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}() >= 1".format(self.get_paddle_api().strip('\n'))
        return code


class FunctionInterpolateMatcher(BaseMatcher):
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
                # TODO: should handle these args specially
                if k in ['recompute_scale_factor', 'antialias']:
                    kwargs.pop(k)
                    continue
                
                #TODO: kwargs_change -> kwargs_mapping
                # not mapping in kwargs in there is not in kwargs_mapping
                new_kwargs[k] = kwargs[k]

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))
        return code.strip('\n')


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
        if 'dilation' in kwargs:
            if kwargs['dilation'] != '(1)':
                return None
            else:
                kwargs.pop('dilation')
        
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
            for key in list(kwargs_change.keys()):
                if key in kwargs:
                    kwargs[kwargs_change[key]] = kwargs[key]
                    kwargs.pop(key)
                    
        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.MaxPool2D({})
            '''
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class DivMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'rounding_mode' in kwargs and kwargs['rounding_mode'] != 'None':
            rounding_mode = kwargs['rounding_mode']
        else:
            rounding_mode = None

        if 'out' in kwargs:
            if rounding_mode is not None and 'trunc' in rounding_mode:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    paddle.assign(paddle.trunc(paddle.divide(x={}, y={})), output={})
                    '''
                )
            elif rounding_mode is not None and 'floor' in rounding_mode:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    paddle.assign(paddle.trunc(paddle.divide(x={}, y={})), output={})
                    '''
                )
            else:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    paddle.assign(paddle.divide(x={}, y={}), output={})
                    '''
                )
            code = API_TEMPLACE.format(kwargs['input'], kwargs['other'], kwargs['out'])
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

        if 'requires_grad' in kwargs:
            stop_gradient = kwargs['requires_grad']
        else:
            stop_gradient = False
        
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
            {}.stop_gradient = not {}
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
        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.where({}, {}, {})
            '''
        )
        code = API_TEMPLACE.format(kwargs['mask'], self.paddleClass, kwargs['value'])
        return ast.parse(code).body


class TensorUniqueMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if 'sorted' in kwargs:
            if 'False' in kwargs['sorted']:
                return None
            else:
                kwargs.pop('sorted')
        
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
            for key in list(kwargs_change.keys()):
                if key in kwargs:
                    kwargs[kwargs_change[key]] = kwargs[key]
                    kwargs.pop(key)

        API_TEMPLACE = textwrap.dedent(
            '''
            {}.unique({})
            '''
        )
        code = API_TEMPLACE.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorExpandMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if 'size' in kwargs:
            kwargs = { 'shape' : kwargs.pop('size'), **kwargs}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip('\n')
            else:
                shape = self.parse_args(args)[0]
            kwargs = { 'shape' : str(shape).replace('\'', ''), **kwargs}

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
            stop_gradient = kwargs['requires_grad']
        else:
            stop_gradient = 'True'

        API_TEMPLACE = textwrap.dedent(
            '''
            {}.stop_gradient = not {}
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

            kwargs['reduction'] = reduction

        if 'target' in kwargs:
            kwargs['label'] = kwargs.pop('target')

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.l1_loss({})
            '''
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
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
            
            kwargs['reduction'] = reduction

        if 'input' in kwargs:
            kwargs['logit'] = kwargs.pop('input')

        if 'target' in kwargs:
            kwargs['label'] = kwargs.pop('target')

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.binary_cross_entropy_with_logits({})
            '''
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class FunctionalMaxPool2DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'dilation' in kwargs:
            if kwargs['dilation'] != '(1)':
                return None
            else:
                kwargs.pop('dilation')
        
        if 'kwargs_change' in self.api_mapping:
            kwargs_change = self.api_mapping['kwargs_change']
            for key in list(kwargs_change.keys()):
                if key in kwargs:
                    kwargs[kwargs_change[key]] = kwargs[key]
                    kwargs.pop(key)
        
        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.functional.max_pool2d({})
            '''
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
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

class WhereMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) == 1:
            code = "paddle.nonzero(x={}, as_tuple=True)".format(kwargs['condition'])
            return code
        else:
            return GenericMatcher.generate_code(self, kwargs)


class TensorIndexCopyMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if kwargs['dim'][0] != '(':
            return None

        count = int(kwargs['dim'][1:-1])

        if count == 0:
            code = '{}.scatter_({}, {})'.format(self.paddleClass, kwargs['index'], kwargs['tensor'])
            return code

        index_list = ['i' + str(i) for i in range(count)]
        tab = '    '
        for_list = ['{}for i{} in range(dim[{}]):'.format(tab * i, str(i), str(i)) for i in range(count)]
        for_body = '\n'.join(for_list)
        exp1 = ','.join(index_list) + ',:'
        exp2 = ','.join([i for i in index_list])

        API_TEMPLATE = textwrap.dedent(
            '''
            dim = list({}.shape)
            {}
            {}{}[{}] = {}[{}].scatter_({}, {}[{}])
            x.clone()
            '''
        )

        code = API_TEMPLATE.format(self.paddleClass,
                                   for_body, tab * count,
                                   self.paddleClass, exp1, self.paddleClass, exp2, kwargs['index'],
                                   kwargs['tensor'], exp2)

        return code


class InstanceNorm3DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if 'eps' not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs['eps']

        if 'momentum' in kwargs:
            momentum = kwargs['momentum']
        else:
            momentum = 0.1

        if 'affine' in kwargs and 'False' in kwargs['affine']:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.InstanceNorm3D(num_features={},
                                    momentum=1-{},
                                    epsilon={},
                                    weight_attr=paddle.ParamAttr(learning_rate=0.0),
                                    bias_attr=paddle.ParamAttr(learning_rate=0.0))
                '''
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                '''
                paddle.nn.InstanceNorm3D(num_features={},
                                    momentum=1-{},
                                    epsilon={},
                                    weight_attr=None,
                                    bias_attr=None)
                '''
            )
        code = API_TEMPLACE.format(kwargs['num_features'], momentum, epsilon)
        return code


class BCEWithLogitsLossMatcher(BaseMatcher):
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

            kwargs['reduction'] = reduction

        API_TEMPLACE = textwrap.dedent(
            '''
            paddle.nn.BCEWithLogitsLoss({})
            '''
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class TensorToMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):

        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)
        if not kwargs:
            code = '{}.cast(dtype = {}.dtype)'.format(self.paddleClass, self.paddleClass)
        elif 'tensor' in kwargs:
            code = '{}.cast(dtype = {}.dtype)'.format(self.paddleClass, kwargs['tensor'])
        elif 'dtype' in kwargs:
            code = '{}.cast(dtype = {})'.format(self.paddleClass, kwargs['dtype'])
        elif 'device' in kwargs and 'dtype' not in kwargs:
            code = '{}.clone()'.format(self.paddleClass)
        else:
            if 'y' not in kwargs:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    if isinstance({}, paddle.dtype):
                        dtype = {}
                    elif isinstance({}, str):
                        dtype = {}.dtype
                    else: 
                        dtype = {}.dtype
                    {}.cast(dtype)
                    '''
                )
                code = API_TEMPLACE.format(kwargs['x'], kwargs['x'],
                                           kwargs['x'], self.paddleClass, kwargs['x'], self.paddleClass)
            else:
                API_TEMPLACE = textwrap.dedent(
                    '''
                    if isinstance({}, paddle.dtype):
                        dtype = {}
                    elif isinstance({}, str):
                        if not isinstance({}, paddle.dtype):
                            dtype = {}.dtype
                        else: 
                            dtype = {}
                    else: 
                        dtype = {}.dtype
                    {}.cast(dtype)'
                    '''
                )
                code = API_TEMPLACE.format(kwargs['x'], kwargs['x'],
                                           kwargs['y'], self.paddleClass, kwargs['y'], kwargs['x'], self.paddleClass)

        return ast.parse(code).body


class GeneratorMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if not kwargs:
            code = 'paddle.fluid.core.default_cpu_generator()'
        elif 'device' in kwargs:
            if kwargs['device'] == '"""cuda"""':
                code = textwrap.dedent(
                    '''
                    device = paddle.device.get_device()
                    paddle.fluid.core.default_cuda_generator(int(device[-1]))
                    '''
                )
            elif kwargs['device'] == '"""mps"""':
                # paddle not suppor mps, but support xpu
                return None

            else:
                code = 'paddle.fluid.core.default_cpu_generator()'

        return code


class TorchUtilDataBatchSampler(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            '''
            sampler = {}
            sampler = sampler if isinstance(sampler, paddle.fluid.dataloader.sampler.Sampler) else paddle.io.Sampler(sampler)            
            paddle.io.BatchSampler(sampler = sampler, batch_size = {}, drop_last = {})
             '''
        )

        code = API_TEMPLATE.format(kwargs['sampler'], kwargs["batch_size"], kwargs["drop_last"])

        return code


class SizeMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        v = astor.to_source(args[0]).strip('\n')


        API_TEMPLATE = textwrap.dedent(
            '''
            list({}.tolist() if isinstance({}, paddle.Tensor) else {})
            '''
        )

        code = API_TEMPLATE.format(v, v ,v )

        node = ast.parse(code.strip('\n')).body
        return node