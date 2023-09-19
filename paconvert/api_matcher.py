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

"""
   isort:skip_file
"""

import ast
import textwrap

import astor

from paconvert.base import BaseMatcher
from paconvert.transformer.custom_op_transformer import CPP_EXTENSION_LIST  # noqa: F401
from paconvert.utils import get_unique_name, process_reduce_and_size_average


class GenericMatcher(BaseMatcher):
    def get_paddle_api(self):
        assert "paddle_api" in self.api_mapping
        if self.paddle_api:
            return self.paddle_api
        return self.api_mapping["paddle_api"]

    def generate_code(self, kwargs):
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
        new_kwargs = {}
        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    # rename/copy in new_kwargs
                    if isinstance(kwargs_change[k], list):
                        for v in kwargs_change[k]:
                            new_kwargs[v] = kwargs[k]
                    else:
                        new_kwargs[kwargs_change[k]] = kwargs[k]
                else:
                    # remove in new_kwargs
                    kwargs.pop(k)
            else:
                # copy to new_kwargs
                new_kwargs[k] = kwargs.pop(k)

                # common process for some args
                if k in [
                    "layout",
                    "device",
                    "memory_format",
                    "inplace",
                    "generator",
                    "non_blocking",
                    "async",
                ]:
                    new_kwargs.pop(k)
                    continue

        new_kwargs = self.set_paddle_default_kwargs(new_kwargs)

        dtype_v = "None"
        if "dtype" in new_kwargs and "dtype" not in kwargs:
            dtype_v = new_kwargs.pop("dtype")

        pin_memory_v = "(False)"
        if "pin_memory" in new_kwargs and "pin_memory" not in kwargs:
            pin_memory_v = new_kwargs.pop("pin_memory")

        stop_gradient_v = "None"
        if "requires_grad" in new_kwargs and "requires_grad" not in kwargs:
            stop_gradient_v = "not " + new_kwargs.pop("requires_grad").strip("()")

        out_v = "None"
        if "out" in new_kwargs and "out" not in kwargs:
            out_v = new_kwargs.pop("out")

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))

        if dtype_v != "None":
            code += ".astype({})".format(dtype_v)

        if pin_memory_v != "(False)":
            code += ".pin_memory()"

        if stop_gradient_v != "None" and out_v != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}, output={})
                {}.stop_gradient = {}
                {}
                """
            )
            code = API_TEMPLATE.format(code, out_v, out_v, stop_gradient_v, out_v)
        elif stop_gradient_v != "None" and out_v == "None":
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(out, code, out, stop_gradient_v, out)
        elif out_v != "None" and stop_gradient_v == "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}, output={})
                """
            )
            code = API_TEMPLATE.format(code, out_v)

        return code


class TensorFunc2PaddleFunc(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]

        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    kwargs[kwargs_change[k]] = kwargs.pop(k)
                else:
                    kwargs.pop(k)
            else:
                if k in [
                    "layout",
                    "device",
                    "memory_format",
                    "inplace",
                    "generator",
                    "non_blocking",
                    "async",
                ]:
                    kwargs.pop(k)

        kwargs = self.set_paddle_default_kwargs(kwargs)

        code = "{}({}, {})".format(
            self.get_paddle_api(), self.paddleClass, self.kwargs_to_str(kwargs)
        )
        return code


class DeleteMatcher(BaseMatcher):
    def get_paddle_api(self):
        return "delete"

    def get_paddle_class_attribute_nodes(self, node):
        return "delete"

    def get_paddle_nodes(self, args, kwargs):
        return "delete"

    def get_paddle_class_nodes(self, func, args, kwargs):
        return "delete"


class UnchangeMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        return "unchange"

    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        if new_kwargs is not None:
            code = "{}({})".format(
                self.get_paddle_api(), self.args_and_kwargs_to_str(new_args, new_kwargs)
            )
            return ast.parse(code).body
        return None

    def get_paddle_class_nodes(self, func, args, kwargs):
        return "unchange"


class SetTrueMatcher(BaseMatcher):
    def get_paddle_api(self):
        return "True"

    def generate_code(self, kwargs):
        return "True"


class SetFalseMatcher(BaseMatcher):
    def get_paddle_api(self):
        return "False"

    def generate_code(self, kwargs):
        return "False"


class InitMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
        for k in kwargs_change:
            if k in kwargs:
                kwargs[kwargs_change[k]] = kwargs.pop(k)

        init_tensor = kwargs.pop("tensor")
        API_TEMPLATE = textwrap.dedent(
            """
            init_{} = {}({})
            init_{}({})
            """
        )
        init_name = self.get_paddle_api().split(".")[-1]
        code = API_TEMPLATE.format(
            init_name,
            self.get_paddle_api(),
            self.kwargs_to_str(kwargs),
            init_name,
            init_tensor,
        )
        return code


class Num2TensorBinaryWithAlphaMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")
        if "other" in kwargs:
            if "alpha" in kwargs:
                kwargs["y"] = "paddle.to_tensor({}*{})".format(
                    kwargs.pop("alpha"), kwargs.pop("other")
                )
            else:
                kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("other"))

        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class TensorInplaceLogicalMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            _input_dtype_ = {}.dtype
            paddle.assign({}({}).cast(_input_dtype_), {})
            """
        )

        # the case of non-inplace op
        # if "input" in kwargs:
        #     kwargs["x"] = kwargs.pop("input")
        if "other" in kwargs:
            kwargs["y"] = kwargs.pop("other")

        code = API_TEMPLATE.format(
            self.paddleClass,
            self.get_paddle_api(),
            self.kwargs_to_str(kwargs),
            self.paddleClass,
        )

        return code


class TensorAddMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def add(self, *args, **kwargs):
                if 'other' in kwargs:
                    y = kwargs['other']
                elif 'y' in kwargs:
                    y = kwargs['y']
                else:
                    y = args[0]

                if 'alpha' in kwargs:
                    alpha = kwargs['alpha']
                    if alpha != 1:
                        if not isinstance(y, paddle.Tensor):
                            y = paddle.to_tensor(alpha * y)
                        else:
                            y = alpha * y
                else:
                    if not isinstance(y, paddle.Tensor):
                        y = paddle.to_tensor(y)

                return paddle.add(self, y)

            setattr(paddle.Tensor, 'add', add)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if len(kwargs) == 0:
            return "unchange"

        self.write_aux_code()
        return "unchange"


class TensorSubtractMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def sub(self, *args, **kwargs):
                if 'other' in kwargs:
                    y = kwargs['other']
                elif 'y' in kwargs:
                    y = kwargs['y']
                else:
                    y = args[0]

                if 'alpha' in kwargs:
                    alpha = kwargs['alpha']
                    if alpha != 1:
                        if not isinstance(y, paddle.Tensor):
                            y = paddle.to_tensor(alpha * y)
                        else:
                            y = alpha * y
                else:
                    if not isinstance(y, paddle.Tensor):
                        y = paddle.to_tensor(y)

                return paddle.subtract(self, y)

            setattr(paddle.Tensor, 'sub', sub)
            setattr(paddle.Tensor, 'subtract', sub)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.write_aux_code()
        return "unchange"


class TensorMultiplyMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def mul(self, *args, **kwargs):
                if 'other' in kwargs:
                    y = kwargs['other']
                elif 'y' in kwargs:
                    y = kwargs['y']
                else:
                    y = args[0]

                if not isinstance(y, paddle.Tensor):
                    y = paddle.to_tensor(y)

                return paddle.multiply(self, y)

            setattr(paddle.Tensor, 'mul', mul)
            setattr(paddle.Tensor, 'multiply', mul)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.write_aux_code()
        return "unchange"


class TensorDivideMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def div(self, *args, **kwargs):
                if 'other' in kwargs:
                    y = kwargs['other']
                elif 'y' in kwargs:
                    y = kwargs['y']
                else:
                    y = args[0]

                if not isinstance(y, paddle.Tensor):
                    y = paddle.to_tensor(y)

                res = paddle.divide(self, y)

                if 'rounding_mode' in kwargs:
                    rounding_mode = kwargs['rounding_mode']
                    if rounding_mode=='trunc':
                        res = paddle.trunc(res)
                    elif rounding_mode=='floor':
                        res = paddle.floor(res)

                return res

            setattr(paddle.Tensor, 'div', div)
            setattr(paddle.Tensor, 'divide', div)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.write_aux_code()
        return "unchange"


class TransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) != 3:
            return None

        API_TEMPLATE = textwrap.dedent(
            """
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose(x=x, perm={})
            """
        )
        perm = get_unique_name("perm")
        code = API_TEMPLATE.format(
            kwargs["input"],
            perm,
            perm,
            kwargs["dim0"],
            kwargs["dim1"],
            perm,
            kwargs["dim1"],
            kwargs["dim0"],
            perm,
        )
        return code


class SwapAxesMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if len(kwargs) != 3:
            return None

        if "dim0" in kwargs:
            kwargs["axis0"] = kwargs.pop("dim0")
            kwargs["axis1"] = kwargs.pop("dim1")

        API_TEMPLATE = textwrap.dedent(
            """
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            paddle.transpose(x=x, perm={})
            """
        )
        perm = get_unique_name("perm")
        code = API_TEMPLATE.format(
            kwargs["input"],
            perm,
            perm,
            kwargs["axis0"],
            kwargs["axis1"],
            perm,
            kwargs["axis1"],
            kwargs["axis0"],
            perm,
        )
        return code


class CreateMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "size" in kwargs:
            kwargs = {"shape": kwargs.pop("size"), **kwargs}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip("\n")
            else:
                shape = self.parse_args(args)[0]

            kwargs = {"shape": str(shape).replace("'", ""), **kwargs}

        for k in ["layout", "device", "memory_format"]:
            if k in kwargs:
                kwargs.pop(k)

        code = GenericMatcher.generate_code(self, kwargs)
        return ast.parse(code).body


class DeviceMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) == 1:
            code = f'str({kwargs["type"]}).replace("cuda", "gpu")'

        if len(kwargs) == 2:
            code = f'":".join([{kwargs["type"]}.replace("cuda", "gpu"), str({kwargs["index"]})])'
        return code


class GeluMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")

        if "approximate" in kwargs:
            approximate_v = kwargs.pop("approximate")
            if "none" in approximate_v:
                kwargs["approximate"] = "False"
            elif "tanh" in approximate_v:
                kwargs["approximate"] = "True"

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class SequentialMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        # nn.Sequential(OrderedDict([...]) / nn.Sequential(OrderedDict(blocks))
        if (
            len(args) == 1
            and isinstance(args[0], ast.Call)
            and self.get_full_attr(args[0].func).endswith("OrderedDict")
        ):
            new_args = self.parse_args(args[0].args)
            new_args = ["*{}".format(new_args[0])]
        # nn.Sequential(module1, module2, ...)
        else:
            new_args = self.parse_args(args)
        code = "paddle.nn.Sequential({})".format(self.args_to_str(new_args))
        return ast.parse(code).body


class PadMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "Reflection" in self.torch_api:
            kwargs["mode"] = "'reflect'"
        elif "Replication" in self.torch_api:
            kwargs["mode"] = "'replicate'"
        elif "Constant" in self.torch_api:
            kwargs["mode"] = "'constant'"
        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class MaxMinMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_kwargs = self.parse_kwargs(kwargs)
        if new_kwargs is None:
            return None

        call_maximinimum = False
        if len(args) > 1 and not isinstance(args[1], ast.Num):
            call_maximinimum = True

        if "other" in new_kwargs:
            call_maximinimum = True

        # the case of two Tensor
        if call_maximinimum:
            return GenericMatcher(
                self.transformer, self.torch_api, self.api_mapping, self.logger
            ).get_paddle_nodes(args, kwargs)

        # the case of one Tensor
        args_list = ["input", "dim", "keepdim", "out"]

        # parse args to kwargs
        for i in range(len(args)):
            new_kwargs[args_list[i]] = astor.to_source(args[i]).strip("\n")
        for node in kwargs:
            new_kwargs[node.arg] = astor.to_source(node.value).strip("\n")

        # change kwargs' name
        if "input" in new_kwargs:
            new_kwargs["x"] = new_kwargs.pop("input")
        if "dim" in new_kwargs:
            new_kwargs["axis"] = new_kwargs.pop("dim")

        paddle_api = self.torch_api.replace("torch", "paddle")
        paddle_api_arg = "paddle.argmin" if "min" in paddle_api else "paddle.argmax"

        if "axis" in new_kwargs and "out" not in new_kwargs:
            return ast.parse(
                "{}({}), {}({})".format(
                    paddle_api,
                    self.kwargs_to_str(new_kwargs),
                    paddle_api_arg,
                    self.kwargs_to_str(new_kwargs),
                )
            ).body
        elif "axis" not in new_kwargs and "out" not in new_kwargs:
            return ast.parse(
                "{}({})".format(paddle_api, self.kwargs_to_str(new_kwargs))
            ).body
        elif "axis" in new_kwargs and "out" in new_kwargs:
            out_v = new_kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}({}), {}[0]), paddle.assign({}({}), {}[1])
                """
            )
            code = API_TEMPLATE.format(
                paddle_api,
                self.kwargs_to_str(new_kwargs),
                out_v,
                paddle_api_arg,
                self.kwargs_to_str(new_kwargs),
                out_v,
            )
            return ast.parse(code.strip("\n")).body
        else:
            out_v = new_kwargs.pop("out")
            return ast.parse(
                "paddle.assign({}({}), {})".format(
                    paddle_api, self.kwargs_to_str(new_kwargs), out_v
                )
            ).body


class EqualMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
        new_kwargs = {}

        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)

        API_TEMPLATE = textwrap.dedent(
            """
            {}({}).item()
            """
        )

        code = API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(new_kwargs)
        )
        return code.strip("\n")


class TensorMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None

        if "size" in kwargs:
            shape = kwargs.pop("size")
        else:
            if len(args) == 0:
                # torch has bug, treat 0D as 0-Size, but paddle not support 0-size
                return None
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip("\n")
            else:
                data = self.parse_args(args)[0]
                if (
                    "torch.IntTensor" == self.torch_api
                    or "torch.cuda.IntTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='int32')".format(data)
                elif ("torch.ShortTensor" == self.torch_api) or (
                    "torch.cuda.ShortTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='int16')".format(data)
                elif (
                    "torch.LongTensor" == self.torch_api
                    or "torch.cuda.LongTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='int64')".format(data)
                elif ("torch.HalfTensor" == self.torch_api) or (
                    "torch.cuda.HalfTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='float16')".format(data)
                elif (
                    "torch.FloatTensor" == self.torch_api
                    or "torch.cuda.FloatTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='float32')".format(data)
                elif ("torch.DoubleTensor" == self.torch_api) or (
                    "torch.cuda.DoubleTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='float64')".format(data)
                elif (
                    "torch.ByteTensor" == self.torch_api
                    or "torch.cuda.ByteTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='uint8')".format(data)
                elif ("torch.BoolTensor" == self.torch_api) or (
                    "torch.cuda.BoolTensor" == self.torch_api
                ):
                    code = "paddle.to_tensor(data={}, dtype='bool')".format(data)

                else:
                    if not isinstance(args[0], ast.Name):
                        code = "paddle.to_tensor(data={}, dtype='float32')".format(data)
                    else:
                        code = "paddle.to_tensor(data={})".format(data)
                node = ast.parse(code.strip("\n")).body
                return node
            shape = str(shape).replace("'", "")

        if (
            "torch.IntTensor" == self.torch_api
            or "torch.cuda.IntTensor" == self.torch_api
        ):
            code = "paddle.empty(shape={}, dtype='int32')".format(shape)
        elif (
            "torch.ShortTensor" == self.torch_api
            or "torch.cuda.ShortTensor" == self.torch_api
        ):
            code = "paddle.empty(shape={}, dtype='int16')".format(shape)
        elif (
            "torch.LongTensor" == self.torch_api
            or "torch.cuda.LongTensor" == self.torch_api
        ):
            code = "paddle.empty(shape={}, dtype='int64')".format(shape)
        elif (
            "torch.HalfTensor" == self.torch_api
            or "torch.cuda.HalfTensor" == self.torch_api
        ):
            code = "paddle.empty(shape={}, dtype='float16')".format(shape)
        elif (
            "torch.FloatTensor" == self.torch_api
            or "torch.cuda.FloatTensor" == self.torch_api
        ):
            code = "paddle.empty(shape={}, dtype='float32')".format(shape)
        elif (
            "torch.DoubleTensor" == self.torch_api
            or "torch.cuda.DoubleTensor" == self.torch_api
        ):
            code = "paddle.empty(shape={}, dtype='float64')".format(shape)
        elif (
            "torch.ByteTensor" == self.torch_api
            or "torch.cuda.ByteTensor" == self.torch_api
        ):
            code = "paddle.zeros(shape={}, dtype='uint8')".format(shape)
        elif ("torch.BoolTensor" == self.torch_api) or (
            "torch.cuda.BoolTensor" == self.torch_api
        ):
            code = "paddle.randint(0, 2, shape={}).astype('bool')".format(shape)
        else:
            code = "paddle.empty(shape={})".format(shape)

        node = ast.parse(code.strip("\n")).body
        return node


class RandintMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "high" in kwargs and "," in kwargs["high"]:
            kwargs["shape"] = kwargs["high"]
            kwargs["high"] = kwargs["low"]
            kwargs["low"] = "0"

        code = GenericMatcher.generate_code(self, kwargs)

        return code


class ScatterMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "async_op" in kwargs:
            kwargs["sync_op"] = f"not {kwargs.pop('async_op')}"
        return GenericMatcher.generate_code(self, kwargs)


class SparseSoftmaxMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = ""
        if "dtype" in kwargs:
            dtype_v = kwargs.pop("dtype")
            tmp_val = get_unique_name("tmp_val")
            code = code + "{}=paddle.sparse.cast({}, value_dtype={})\n".format(
                tmp_val, kwargs["input"], dtype_v
            )
            kwargs["input"] = tmp_val
        code = code + GenericMatcher.generate_code(self, kwargs)
        return code


class TensorTransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        # may be ndarray.transpose([list]) / ndarray.transpose(list)
        if len(kwargs) != 2:
            return "misidentify"

        API_TEMPLATE = textwrap.dedent(
            """
            x = {}
            {} = list(range(x.ndim))
            {}[{}] = {}
            {}[{}] = {}
            x.transpose(perm={})
            """
        )
        perm = get_unique_name("perm")
        code = API_TEMPLATE.format(
            self.paddleClass,
            perm,
            perm,
            kwargs["dim0"],
            kwargs["dim1"],
            perm,
            kwargs["dim1"],
            kwargs["dim0"],
            perm,
        )
        return code


class TensorSizeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" in kwargs:
            code = "{}.shape[{}]".format(self.paddleClass, kwargs["dim"])
        else:
            code = "{}.shape".format(self.paddleClass)
        return code


class TensorPermuteMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)

        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            perm_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            perm_list = self.parse_args(args)

        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None

        if "dims" in kwargs:
            kwargs = {"perm": kwargs.pop("dims"), **kwargs}
        else:
            kwargs = {"perm": str(perm_list).replace("'", ""), **kwargs}

        code = "{}.transpose({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorRenameMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "columns" in kwargs:
            return "misidentify"

        return None


class TensorRepeatMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def repeat(self, *args, **kwargs):
                if args:
                    if len(args)==1 and isinstance(args[0], (tuple, list)):
                        return paddle.tile(self, args[0])
                    else:
                        return paddle.tile(self, list(args))
                elif kwargs:
                    assert 'repeats' in kwargs
                    return paddle.tile(self, repeat_times=kwargs['repeats'])

            setattr(paddle.Tensor, 'repeat', repeat)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if kwargs:
            if len(kwargs) == 1 and "repeats" in kwargs:
                self.write_aux_code()
                return "unchange"
            else:
                return "misidentify"

        if args:
            self.write_aux_code()
            return "unchange"

        return "misidentify"


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
        code = f"{self.paddleClass}.astype(dtype={kwargs['tensor']}.dtype)"
        return code


class TensorNew_Matcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "size" in kwargs:
            kwargs = {"shape": kwargs.pop("size"), **kwargs}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip("\n")
            else:
                shape = self.parse_args(args)[0]

            kwargs = {"shape": str(shape).replace("'", ""), **kwargs}

        for k in ["layout", "device", "memory_format"]:
            if k in kwargs:
                kwargs.pop(k)

        stop_gradient_v = None
        if "requires_grad" in kwargs:
            stop_gradient_v = "not " + kwargs.pop("requires_grad").strip("()")

        pin_memory_v = False
        if "pin_memory" in kwargs and kwargs.pop("pin_memory") == "(True)":
            pin_memory_v = True

        if "dtype" not in kwargs:
            kwargs["dtype"] = "{}.dtype".format(self.paddleClass)

        if stop_gradient_v:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}({})
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                out,
                self.get_paddle_api(),
                self.kwargs_to_str(kwargs),
                out,
                stop_gradient_v,
                out,
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        if pin_memory_v:
            code = code.rstrip("\n") + ".pin_memory()"

        return ast.parse(code.strip("\n")).body


class TensorNewFullMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = {"shape": kwargs.pop("size"), **kwargs}
        for k in ["layout", "device", "memory_format"]:
            if k in kwargs:
                kwargs.pop(k)

        stop_gradient_v = None
        if "requires_grad" in kwargs:
            stop_gradient_v = "not " + kwargs.pop("requires_grad").strip("()")

        pin_memory_v = False
        if "pin_memory" in kwargs and kwargs.pop("pin_memory") == "(True)":
            pin_memory_v = True

        if "dtype" not in kwargs:
            kwargs["dtype"] = "{}.dtype".format(self.paddleClass)

        if stop_gradient_v:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = paddle.full({})
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                out, self.kwargs_to_str(kwargs), out, stop_gradient_v, out
            )
        else:
            code = "paddle.full({})".format(self.kwargs_to_str(kwargs))

        if pin_memory_v:
            code = code.rstrip("\n") + ".pin_memory()"

        return code.strip("\n")


class TensorNewTensorMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "layout" in kwargs:
            kwargs.pop("layout")

        if "device" in kwargs:
            kwargs.pop("device")

        if "requires_grad" in kwargs:
            kwargs["stop_gradient"] = "not " + kwargs.pop("requires_grad").strip("()")

        if "pin_memory" in kwargs and kwargs.pop("pin_memory") == "(True)":
            kwargs["place"] = "paddle.CUDAPinnedPlace()"

        if "dtype" not in kwargs:
            kwargs["dtype"] = "{}.dtype".format(self.paddleClass)

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code.strip("\n")


class TorchTensorMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "device" in kwargs:
            kwargs["place"] = kwargs.pop("device")

        if "requires_grad" in kwargs:
            kwargs["stop_gradient"] = "not " + kwargs.pop("requires_grad").strip("()")

        if "pin_memory" in kwargs and kwargs.pop("pin_memory") == "(True)":
            kwargs["place"] = "paddle.CUDAPinnedPlace()"

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code.strip("\n")


class CudaIsAvailableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}() >= 1".format(self.get_paddle_api().strip("\n"))
        return code


class CudnnIsAvailableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "bool(paddle.device.get_cudnn_version())"
        return code


class BatchNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "momentum" in kwargs:
            kwargs["momentum"] = f"1 - {kwargs.pop('momentum')}"

        if "affine" in kwargs:
            kwargs["weight_attr"] = f"None if {kwargs['affine']} else False"
            kwargs["bias_attr"] = f"None if {kwargs.pop('affine')} else False"

        if "track_running_stats" in kwargs:
            kwargs["use_global_stats"] = f"not {kwargs.pop('track_running_stats')}"

        code = GenericMatcher.generate_code(self, kwargs)
        return code


class SplitMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def split(x, num_or_sections, axis=0):
                if isinstance(num_or_sections, int):
                    return paddle.split(x, x.shape[axis]//num_or_sections, axis)
                else:
                    return paddle.split(x, num_or_sections, axis)

            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.write_aux_code()
        return GenericMatcher.generate_code(self, kwargs).replace(
            "paddle", "paddle_aux"
        )


class RangeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        else:
            dtype = '"""float32"""'

        if "requires_grad" in kwargs:
            stop_gradient = kwargs["requires_grad"]
        else:
            stop_gradient = False

        if "start" in kwargs:
            start = kwargs["start"]
        else:
            start = 0

        if "step" in kwargs:
            step = kwargs["step"]
        else:
            step = 1

        out = get_unique_name("out")
        API_TEMPLATE = textwrap.dedent(
            """
            {} = paddle.arange(start={}, end={}+{} if ({} - {}) % {} == 0 else {}, step={}, dtype={})
            {}.stop_gradient = not {}
            {}
            """
        )
        code = API_TEMPLATE.format(
            out,
            start,
            kwargs["end"],
            step,
            kwargs["end"],
            start,
            step,
            kwargs["end"],
            step,
            dtype,
            out,
            stop_gradient,
            out,
        )
        return code


class MeshgridMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        if new_kwargs is None:
            return None
        if "indexing" in new_kwargs and "ij" not in new_kwargs["indexing"]:
            code = "list([i.T for i in {}({})])".format(
                self.get_paddle_api(), self.args_to_str(new_args)
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.args_to_str(new_args))
        return ast.parse(code).body


class TensorSkipMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        code = "{}".format(self.paddleClass)
        return ast.parse(code).body


class TensorCopyMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        args = self.parse_args(args)
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.assign({}, output={})
            """
        )
        code = API_TEMPLATE.format(args[0], self.paddleClass)
        return ast.parse(code).body


class TensorMaskedFillMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.where({}, {}, {})
            """
        )
        code = API_TEMPLATE.format(kwargs["mask"], self.paddleClass, kwargs["value"])
        return ast.parse(code).body


class TensorExpandMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "size" in kwargs:
            kwargs = {"shape": kwargs.pop("size"), **kwargs}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).strip("\n")
            else:
                shape = self.parse_args(args)[0]
            kwargs = {"shape": str(shape).replace("'", ""), **kwargs}

        code = "{}.expand({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorSoftmaxMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        else:
            return None

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.softmax({}, {})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorRequiresGrad_Matcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if "requires_grad" in kwargs:
            requires_grad_v = kwargs["requires_grad"]
        else:
            requires_grad_v = "True"

        API_TEMPLATE = textwrap.dedent(
            """
            {} = {}
            {}.stop_gradient = not {}
            {}
            """
        )
        out = get_unique_name("out")
        code = API_TEMPLATE.format(out, self.paddleClass, out, requires_grad_v, out)
        return ast.parse(code.strip("\n")).body


class LoadMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        unsupported_params = [
            "map_location",
            "pickle_module",
            "weights_only",
            "pickle_load_args",
        ]
        for param in unsupported_params:
            if param in kwargs:
                kwargs.pop(param)

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.load(path={})
            """
        )
        code = API_TEMPLATE.format(kwargs["f"])
        return code


class TensorTypeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) == 0:
            return None
        code = f"{self.paddleClass}.astype({kwargs['dtype']})"
        return code


class TensorIsCudaMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "'gpu' in str({}.place)".format(self.paddleClass)
        return ast.parse(code).body


class SeedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.get_rng_state()[0].current_seed()
            """
        )
        return API_TEMPLATE


class CudaSeedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.get_cuda_rng_state()[0].current_seed()
            """
        )
        return API_TEMPLATE


class SetPrintOptionsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "profile" in kwargs:
            pro_kwargs = {}
            if kwargs["profile"] == '"""default"""':
                pro_kwargs["precision"] = 4
                pro_kwargs["threshold"] = 1000
                pro_kwargs["edgeitems"] = 3
                pro_kwargs["linewidth"] = 80
            elif kwargs["profile"] == '"""short"""':
                pro_kwargs["precision"] = 2
                pro_kwargs["threshold"] = 1000
                pro_kwargs["edgeitems"] = 2
                pro_kwargs["linewidth"] = 80
            elif kwargs["profile"] == '"""full"""':
                pro_kwargs["precision"] = 4
                pro_kwargs["threshold"] = 1000000
                pro_kwargs["edgeitems"] = 3
                pro_kwargs["linewidth"] = 80

            for k in pro_kwargs.keys():
                if k not in kwargs.keys():
                    kwargs[k] = pro_kwargs[k]

            kwargs.pop("profile")

        kwargs = self.set_paddle_default_kwargs(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.set_printoptions({})
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class RandLikeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        stop_gradient_v = None
        if "requires_grad" in kwargs:
            stop_gradient_v = "not " + kwargs["requires_grad"]

        if "dtype" in kwargs and "requires_grad" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}(shape={}.shape, dtype={})
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                out,
                self.get_paddle_api(),
                kwargs["input"],
                kwargs["dtype"],
                out,
                stop_gradient_v,
                out,
            )
        elif "dtype" not in kwargs and "requires_grad" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}(shape={}.shape, dtype={}.dtype)
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                out,
                self.get_paddle_api(),
                kwargs["input"],
                kwargs["input"],
                out,
                stop_gradient_v,
                out,
            )
        elif "dtype" in kwargs and "requires_grad" not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                {}(shape={}.shape, dtype={})
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(), kwargs["input"], kwargs["dtype"]
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {}(shape={}.shape, dtype={}.dtype)
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(), kwargs["input"], kwargs["input"]
            )

        return code


class PolarMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.complex({} * paddle.cos({}), {} * paddle.sin({})), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["abs"],
                kwargs["angle"],
                kwargs["abs"],
                kwargs["angle"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.complex({} * paddle.cos({}), {} * paddle.sin({}))
                """
            )
            code = API_TEMPLATE.format(
                kwargs["abs"], kwargs["angle"], kwargs["abs"], kwargs["angle"]
            )

        return code


class NarrowMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            {} = ({}.shape[{}] + {}) if {} < 0 else {}
            paddle.slice({}, [{}], [{}], [{} + {}])
            """
        )
        start = get_unique_name("start")
        code = API_TEMPLATE.format(
            start,
            kwargs["input"],
            kwargs["dim"],
            kwargs["start"],
            kwargs["start"],
            kwargs["start"],
            kwargs["input"],
            kwargs["dim"],
            start,
            start,
            kwargs["length"],
        )
        return code


class NarrowCopyMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" not in kwargs:
            kwargs["out"] = None

        API_TEMPLATE = textwrap.dedent(
            """
            {} = ({}.shape[{}] + {}) if {} < 0 else {}
            paddle.assign(paddle.slice({}, [{}], [{}], [{} + {}]), {})
            """
        )
        start = get_unique_name("start")
        code = API_TEMPLATE.format(
            start,
            kwargs["input"],
            kwargs["dim"],
            kwargs["start"],
            kwargs["start"],
            kwargs["start"],
            kwargs["input"],
            kwargs["dim"],
            start,
            start,
            kwargs["length"],
            kwargs["out"],
        )
        return code


class AddCMulMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "value" not in kwargs:
            kwargs["value"] = 1

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({} + {} * {} * {}, output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["value"],
                kwargs["tensor1"],
                kwargs["tensor2"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.add({}, {} * {} * {})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["value"], kwargs["tensor1"], kwargs["tensor2"]
            )

        return code


class AddCDivMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "value" not in kwargs:
            kwargs["value"] = 1

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({} + {} * {} / {}, output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["value"],
                kwargs["tensor1"],
                kwargs["tensor2"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.add({}, {} * {} / {})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["value"], kwargs["tensor1"], kwargs["tensor2"]
            )

        return code


class IsNonzeroMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            {}.astype('bool').item()
            """
        )
        code = API_TEMPLATE.format(kwargs["input"])

        return code


class VStackMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                if {}[0].ndim == 1:
                    {} = paddle.stack({})
                else:
                    {} = paddle.concat({})
                paddle.assign({}, output={})
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                kwargs["tensors"],
                out,
                kwargs["tensors"],
                out,
                kwargs["tensors"],
                out,
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                if {}[0].ndim == 1:
                    {} = paddle.stack({})
                else:
                    {} = paddle.concat({})
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                kwargs["tensors"], out, kwargs["tensors"], out, kwargs["tensors"], out
            )

        return code


class HStackMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                {} = 0 if {}[0].ndim == 1 else 1
                paddle.assign(paddle.concat({}, axis={}), output={})
                """
            )
            axis = get_unique_name("axis")
            code = API_TEMPLATE.format(
                axis, kwargs["tensors"], kwargs["tensors"], axis, kwargs["out"]
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = 0 if {}[0].ndim == 1 else 1
                paddle.concat({}, axis={})
                """
            )
            axis = get_unique_name("axis")
            code = API_TEMPLATE.format(axis, kwargs["tensors"], kwargs["tensors"], axis)

        return code


class ColumnStackMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                if {}[0].ndim == 1:
                    {} = paddle.stack({}, axis=1)
                else:
                    {} = paddle.concat({}, axis=1)
                paddle.assign({}, output={})
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                kwargs["tensors"],
                out,
                kwargs["tensors"],
                out,
                kwargs["tensors"],
                out,
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                if {}[0].ndim == 1:
                    {} = paddle.stack({}, axis=1)
                else:
                    {} = paddle.concat({}, axis=1)
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                kwargs["tensors"], out, kwargs["tensors"], out, kwargs["tensors"], out
            )

        return code


# will implenment by aux_code
class TensorIndexCopyMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if kwargs["dim"][1:-1].isdigit() and int(kwargs["dim"][1:-1]) == 0:
            code = "{}.scatter_({}, {})".format(
                self.paddleClass, kwargs["index"], kwargs["tensor"]
            )
            return code

        API_TEMPLATE = textwrap.dedent(
            """
            times, temp_shape, temp_index = paddle.prod(paddle.to_tensor({}.shape[:{}])), {}.shape, {}
            {}, new_t = {}.reshape([-1] + temp_shape[{}+1:]), {}.reshape([-1] + temp_shape[{}+1:])
            for i in range(1, times):
                temp_index= paddle.concat([temp_index, index+len(index)*i])
            {}.scatter_(temp_index, new_t).reshape(temp_shape)
            """
        )

        code = API_TEMPLATE.format(
            self.paddleClass,
            kwargs["dim"],
            self.paddleClass,
            kwargs["index"],
            self.paddleClass,
            self.paddleClass,
            kwargs["dim"],
            kwargs["tensor"],
            kwargs["dim"],
            self.paddleClass,
        )

        return code


class IndexCopyMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        out_v = None
        if "out" in kwargs:
            out_v = kwargs.pop("out")

        if kwargs["dim"][1:-1].isdigit() and int(kwargs["dim"][1:-1]) == 0:
            if out_v:
                code = "paddle.assign(paddle.scatter({}, {}, {}), {})".format(
                    kwargs["input"], kwargs["index"], kwargs["source"], out_v
                )
            else:
                code = "paddle.scatter({}, {}, {})".format(
                    kwargs["input"], kwargs["index"], kwargs["source"]
                )
            return code

        if out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                times, temp_shape, temp_index = paddle.prod(paddle.to_tensor({}.shape[:{}])), {}.shape, {}
                {}, new_t = {}.reshape([-1] + temp_shape[{}+1:]), {}.reshape([-1] + temp_shape[{}+1:])
                for i in range(1, times):
                    temp_index= paddle.concat([temp_index, index+len(index)*i])
                paddle.assign(paddle.scatter({}, temp_index, new_t).reshape(temp_shape), {})
                """
            )

            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["input"],
                kwargs["index"],
                kwargs["input"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["source"],
                kwargs["dim"],
                kwargs["input"],
                out_v,
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                times, temp_shape, temp_index = paddle.prod(paddle.to_tensor({}.shape[:{}])), {}.shape, {}
                {}, new_t = {}.reshape([-1] + temp_shape[{}+1:]), {}.reshape([-1] + temp_shape[{}+1:])
                for i in range(1, times):
                    temp_index= paddle.concat([temp_index, index+len(index)*i])
                paddle.scatter({}, temp_index, new_t).reshape(temp_shape)
                """
            )

            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["input"],
                kwargs["index"],
                kwargs["input"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["source"],
                kwargs["dim"],
                kwargs["input"],
            )

        return code


class InstanceNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "momentum" in kwargs:
            momentum = kwargs["momentum"]
        else:
            momentum = 0.1
        kwargs["momentum"] = f"1-{momentum}"
        return GenericMatcher.generate_code(self, kwargs)


class GeneratorMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if not kwargs:
            code = "paddle.framework.core.default_cpu_generator()"
        elif "device" in kwargs:
            if kwargs["device"] == '"""cuda"""':
                code = textwrap.dedent(
                    """
                    device = paddle.device.get_device()
                    paddle.framework.core.default_cuda_generator(int(device[-1]))
                    """
                )
            elif kwargs["device"] == '"""mps"""':
                # paddle not suppor mps, but support xpu
                return None

            else:
                code = "paddle.framework.core.default_cpu_generator()"

        return code


class RandomSamplerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if (
            kwargs.get("replacement") == "(False)"
            and kwargs.get("num_samples") != "None"
        ):
            return None
        return GenericMatcher.generate_code(self, kwargs)


class SizeMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args) == 0:
            code = "list([])"
        else:
            code = "list({})".format(astor.to_source(args[0]).strip("\n"))

        node = ast.parse(code.strip("\n")).body
        return node


class TensorToMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def to(self, *args, **kwargs):
                args_list = ["x", "y", "non_blocking", "copy", "memory_format"]
                new_kwargs = {}
                for i, node in enumerate(args):
                    k = args_list[i]
                    new_kwargs[k] = node
                for node in kwargs:
                    v = kwargs[node]
                    new_kwargs[node] = v
                kwargs = new_kwargs
                if not kwargs:
                    return self
                elif "tensor" in kwargs:
                    return paddle.cast(self, "{}.dtype".format(kwargs["tensor"]))
                elif "dtype" in kwargs:
                    return paddle.cast(self, "{}".format(kwargs["dtype"]))
                elif "device" in kwargs and "dtype" not in kwargs:
                    return self
                elif kwargs:
                    if "y" not in kwargs and "x" in kwargs:
                        if isinstance(kwargs["x"], paddle.dtype):
                            dtype = kwargs["x"]
                        elif isinstance(kwargs["x"], str) and kwargs["x"] not in ['cpu', 'cuda', 'ipu', 'xpu']:
                            dtype = kwargs["x"]
                        elif isinstance(kwargs["x"], paddle.Tensor):
                            dtype = kwargs["x"].dtype
                        else:
                            dtype = self.dtype
                        return paddle.cast(self, dtype)

                    elif "y" in kwargs and "x" in kwargs:
                        if isinstance(kwargs["x"], paddle.dtype):
                            dtype = kwargs["x"]
                        elif isinstance(kwargs["x"], str):
                            if x not in ['cpu', 'cuda', 'ipu', 'xpu']:
                                dtype = kwargs["x"]
                            else:
                                dtype = kwargs["y"] if isinstance(kwargs["y"], str) else self.dtype
                        else:
                            dtype = kwargs["x"]
                        return paddle.cast(self, dtype)
                    else:
                        return self

            setattr(paddle.Tensor, 'to', to)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.write_aux_code()
        return "unchange"


class TensorRequiresGradMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "not {}.stop_gradient".format(self.paddleClass)
        return ast.parse(code).body


class ArangeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "end" not in kwargs:
            kwargs["end"] = kwargs.pop("start")
        code = GenericMatcher.generate_code(self, kwargs)
        return code


class ErfCMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(1. - paddle.erf({}), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                1. - paddle.erf({})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"])

        return code


class SpecialErfcxMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.exp({} ** 2) * (1.0 - paddle.erf({})), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["input"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.exp({} ** 2) * (1.0 - paddle.erf({}))
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["input"])

        return code


class SpecialXlogyMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({} * paddle.log({}), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {} * paddle.log({})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        return code


class Exp2Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(2. ** {}, output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                2. ** {}
                """
            )
            code = API_TEMPLATE.format(kwargs["input"])

        return code


class ExpitMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        exp_v = get_unique_name("exp_v")
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                {} = paddle.exp({})
                paddle.assign({} / (1 + {}), output={})
                """
            )
            code = API_TEMPLATE.format(
                exp_v, kwargs["input"], exp_v, exp_v, kwargs["out"]
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = paddle.exp({})
                {} / (1 + {})
                """
            )
            code = API_TEMPLATE.format(exp_v, kwargs["input"], exp_v, exp_v)

        return code


class FModMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.mod({}, paddle.to_tensor({}, dtype={}.dtype)), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["other"], kwargs["input"], kwargs["out"]
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.mod({}, paddle.to_tensor({}, dtype={}.dtype))
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["other"], kwargs["input"]
            )

        return code


class LdExpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({} * (2. ** {}), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {} * (2. ** {})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        return code


class LogAddExpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.log(paddle.exp({}) + paddle.exp({})), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.log(paddle.exp({}) + paddle.exp({}))
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        return code


class LogAddExp2Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.log2(2. ** {} + 2. ** {}), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.log2(2. ** {} + 2. ** {})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        return code


class XLogYMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.multiply(paddle.to_tensor({}), paddle.log(paddle.to_tensor({}))), output={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.multiply(paddle.to_tensor({}), paddle.log(paddle.to_tensor({})))
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        return code


class StdMeanMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "correction" in kwargs:
            kwargs["unbiased"] = kwargs.pop("correction")
        elif "unbiased" in kwargs:
            # do nothing
            pass
        else:
            kwargs["unbiased"] = True

        if "keepdim" not in kwargs:
            kwargs["keepdim"] = False

        if "dim" not in kwargs:
            kwargs["dim"] = None

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.std({}, axis={}, unbiased={}, keepdim={}), output={}[0])
                paddle.assign(paddle.mean({}, axis={}, keepdim={}), output={}[1])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["unbiased"],
                kwargs["keepdim"],
                kwargs["out"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                tuple([paddle.std({}, axis={}, unbiased={}, keepdim={}), paddle.mean({}, axis={}, keepdim={})])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["unbiased"],
                kwargs["keepdim"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
            )

        return code


class VarMeanMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "correction" in kwargs:
            kwargs["unbiased"] = kwargs.pop("correction")
        elif "unbiased" in kwargs:
            # do nothing
            pass
        else:
            kwargs["unbiased"] = True

        if "keepdim" not in kwargs:
            kwargs["keepdim"] = False

        if "dim" not in kwargs:
            kwargs["dim"] = None

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.var({}, axis={}, unbiased={}, keepdim={}), output={}[0])
                paddle.assign(paddle.mean({}, axis={}, keepdim={}), output={}[1])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["unbiased"],
                kwargs["keepdim"],
                kwargs["out"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                tuple([paddle.var({}, axis={}, unbiased={}, keepdim={}), paddle.mean({}, axis={}, keepdim={})])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["unbiased"],
                kwargs["keepdim"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
            )

        return code


class AddMRMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        params1 = ["mat1", "mat", "vec1", "batch1"]
        params2 = ["mat2", "vec", "vec2", "batch2"]
        param1, param2 = None, None
        for i, param in enumerate(params1):
            if param in kwargs:
                param1 = kwargs[params1[i]]
                param2 = kwargs[params2[i]]

        if "beta" not in kwargs:
            kwargs["beta"] = 1

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.add({}*{}, {}*{}({}, {})), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["beta"],
                kwargs["input"],
                kwargs["alpha"],
                self.get_paddle_api(),
                param1,
                param2,
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.add({}*{}, {}*{}({}, {}))
                """
            )
            code = API_TEMPLATE.format(
                kwargs["beta"],
                kwargs["input"],
                kwargs["alpha"],
                self.get_paddle_api(),
                param1,
                param2,
            )

        return code


class AddBmmMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "beta" not in kwargs:
            kwargs["beta"] = 1

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.add({}*{}, {}*paddle.sum(paddle.bmm({}, {}), axis=0)), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["beta"],
                kwargs["input"],
                kwargs["alpha"],
                kwargs["batch1"],
                kwargs["batch2"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.add({}*{}, {}*paddle.sum(paddle.bmm({}, {}), axis=0))
                """
            )
            code = API_TEMPLATE.format(
                kwargs["beta"],
                kwargs["input"],
                kwargs["alpha"],
                kwargs["batch1"],
                kwargs["batch2"],
            )

        return code


class CholeskyInverseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "upper" not in kwargs:
            kwargs["upper"] = False

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                {} = list(range({}.ndim))
                {}[-1], {}[-2] = {}[-2], {}[-1]
                if {}:
                    {} = paddle.linalg.inv(paddle.transpose({}, perm={}) @ {})
                else:
                    {} = paddle.linalg.inv({} @ paddle.transpose({}, perm={}))
                paddle.assign({}, output={})
                """
            )
            perm = get_unique_name("perm")
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                perm,
                kwargs["input"],
                perm,
                perm,
                perm,
                perm,
                kwargs["upper"],
                out,
                kwargs["input"],
                perm,
                kwargs["input"],
                out,
                kwargs["input"],
                kwargs["input"],
                perm,
                out,
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = list(range({}.ndim))
                {}[-1], {}[-2] = {}[-2], {}[-1]
                if {}:
                    {} = paddle.linalg.inv(paddle.transpose({}, perm={}) @ {})
                else:
                    {} = paddle.linalg.inv({} @ paddle.transpose({}, perm={}))
                {}
                """
            )
            perm = get_unique_name("perm")
            out = get_unique_name("out")
            code = API_TEMPLATE.format(
                perm,
                kwargs["input"],
                perm,
                perm,
                perm,
                perm,
                kwargs["upper"],
                out,
                kwargs["input"],
                perm,
                kwargs["input"],
                out,
                kwargs["input"],
                kwargs["input"],
                perm,
                out,
            )

        return code


class LogDetMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.log(paddle.linalg.det({}))
            """
        )
        code = API_TEMPLATE.format(kwargs["input"])

        return code


class AvgPoolMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")

        if "count_include_pad" in kwargs:
            kwargs["exclusive"] = "not" + kwargs.pop("count_include_pad")
        else:
            kwargs["exclusive"] = "False"
        API_TEMPLATE = textwrap.dedent(
            """
            {}({})
            """
        )
        code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class FSoftMinMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" not in kwargs or kwargs["dim"] == "None":
            return None

        if "dtype" in kwargs and kwargs["dim"] is not None:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.nn.functional.softmax(-{}, axis={}).astype({})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["dim"], kwargs["dtype"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.nn.functional.softmax(-{}, axis={})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["dim"])

        return code


class FBatchNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["x"] = kwargs.pop("input")
        if "eps" in kwargs:
            kwargs["epsilon"] = kwargs.pop("eps")
        if "momentum" in kwargs:
            kwargs["momentum"] = "1 - " + kwargs["momentum"]

        API_TEMPLATE = textwrap.dedent(
            """
            {}({})
            """
        )
        code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class FInstanceNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["x"] = kwargs.pop("input")
        if "momentum" in kwargs:
            kwargs["momentum"] = "1 - " + kwargs["momentum"]

        API_TEMPLATE = textwrap.dedent(
            """
            {}({})
            """
        )
        code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class MSortMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.sort({}, axis=0), {})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"], kwargs["out"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.sort({}, axis=0)
                """
            )
            code = API_TEMPLATE.format(kwargs["input"])
        return code


class NumelMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass
        return "{}.size".format(kwargs["input"])


class TriangularSolveMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        out_v = kwargs.pop("out") if "out" in kwargs else None
        new_kwargs = {}
        new_kwargs["x"] = kwargs.pop("A")
        new_kwargs["y"] = kwargs.pop("b")
        new_kwargs.update(kwargs)

        if out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(paddle.linalg.triangular_solve({}), {}[0]), paddle.assign({}, {}[1])
                """
            )
            code = API_TEMPLATE.format(
                self.kwargs_to_str(new_kwargs), out_v, new_kwargs["x"], out_v
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.linalg.triangular_solve({}), {}
                """
            )
            code = API_TEMPLATE.format(self.kwargs_to_str(new_kwargs), new_kwargs["x"])

        return code


class IndexAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1.0

        API_TEMPLATE = textwrap.dedent(
            """
            {}({}, index={}, axis={}, value={} * {})
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(),
            kwargs["input"],
            kwargs["index"],
            kwargs["dim"],
            kwargs["alpha"],
            kwargs["source"],
        )

        if "out" in kwargs and kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])
        return code


class AMinMaxMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "dim" not in kwargs:
            kwargs["dim"] = None

        if "keepdim" not in kwargs:
            kwargs["keepdim"] = False

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                tuple([paddle.assign(paddle.amin({}, axis={}, keepdim={}), {}[0]), paddle.assign(paddle.amax({}, axis={}, keepdim={}), {}[1])])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
                kwargs["out"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
                kwargs["out"],
            )

        else:
            API_TEMPLATE = textwrap.dedent(
                """
                tuple([paddle.amin({}, axis={}, keepdim={}), paddle.max({}, axis={}, keepdim={})])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
            )

        return code


class DivideMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input").strip("\n")
        if "other" in kwargs:
            kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("other").strip("\n"))

        rounding_mode_v = "None"
        if "rounding_mode" in kwargs:
            rounding_mode_v = kwargs.pop("rounding_mode")

        out_v = "None"
        if "out" in kwargs:
            out_v = kwargs.pop("out")

        API_TEMPLATE = textwrap.dedent(
            """
            {}({})
            """
        )
        code = API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        if rounding_mode_v != "None":
            if "trunc" in rounding_mode_v:
                code = "paddle.trunc({})".format(code)
            elif "floor" in rounding_mode_v:
                code = "paddle.floor({})".format(code)

        if out_v != "None":
            code = "paddle.assign({}, output={})".format(code, out_v)

        return code


class AllcloseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = GenericMatcher.generate_code(self, kwargs)
        code = code.strip("\n") + ".item()"
        return code


class Assert_AllcloseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["x"], kwargs["y"] = kwargs.pop("actual"), kwargs.pop("expected")
        msg = "''"
        if "msg" in kwargs:
            msg = kwargs.pop("msg")
        code = "assert paddle.allclose({}).item(), {}".format(
            self.kwargs_to_str(kwargs), msg
        )
        return code


class Num2TensorBinaryMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")
        if "other" in kwargs:
            kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("other"))

        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class Chain_MatmulMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args) == 1 and isinstance(args[0], ast.Starred):
            return None
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        if new_kwargs is None:
            return None
        code = "{}".format(new_args[0])
        for arg in new_args[1:]:
            code = code + " @ {}".format(arg)
        if "out" in new_kwargs and new_kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, new_kwargs["out"])

        return ast.parse(code).body


class HypotMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.pow({}**2 + {}**2, 1/2)
            """
        )
        code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        if "out" in kwargs and kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class TensorReshapeMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def reshape(self, *args, **kwargs):
                if args:
                    if len(args)==1 and isinstance(args[0], (tuple, list)):
                        return paddle.reshape(self, args[0])
                    else:
                        return paddle.reshape(self, list(args))
                elif kwargs:
                    assert 'shape' in kwargs
                    return paddle.reshape(self, shape=kwargs['shape'])

            setattr(paddle.Tensor, 'reshape', reshape)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if kwargs:
            if len(kwargs) == 1 and "shape" in kwargs:
                return "unchange"
            else:
                return "misidentify"

        if args:
            if len(args) > 1 and isinstance(args[0], (ast.Tuple, ast.List)):
                return "unchange"
            else:
                self.write_aux_code()
                return "unchange"

        return "misidentify"


class TensorReshape_asMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            {}.reshape({}.shape)
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["other"])

        return code


class TensorResize_as_Matcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            {}.reshape_({}.shape)
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["the_template"])
        return code


class SelectMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if len(kwargs) != 3:
            return "misidentify"

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.index_select({}, index=paddle.to_tensor([{}]), axis={}).squeeze({})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"], kwargs["index"], kwargs["dim"], kwargs["dim"]
        )

        return code


class SearchsortedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "side" in kwargs:
            kwargs["right"] = f"{kwargs.pop('side')} == 'right'"

        if "sorter" in kwargs and kwargs["sorter"] != "None":
            kwargs[
                "sorted_sequence"
            ] = f"{kwargs['sorted_sequence']}.take_along_axis(axis=-1, indices={kwargs.pop('sorter')})"

        code = "paddle.searchsorted({})".format(self.kwargs_to_str(kwargs))
        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            code = "paddle.assign(paddle.searchsorted({}), output={})".format(
                self.kwargs_to_str(kwargs), out_v
            )

        return code


class SincMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            API_TEMPLATE = textwrap.dedent(
                """
                import numpy
                paddle.assign(paddle.where({}==0, x=paddle.to_tensor([1.], dtype={}.dtype), y=paddle.sin(numpy.pi * {}) / (numpy.pi * {})), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["input"],
                kwargs["input"],
                kwargs["input"],
                kwargs["out"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                import numpy
                paddle.where({}==0, x=paddle.to_tensor([1.], dtype={}.dtype), y=paddle.sin(numpy.pi * {}) / (numpy.pi * {}))
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["input"], kwargs["input"], kwargs["input"]
            )

        return code


class SLogDetMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            x_v = kwargs.pop("input")
        elif "A" in kwargs:
            x_v = kwargs.pop("A")
        else:
            x_v = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                res = paddle.linalg.slogdet({})
                paddle.assign(res[0], {}[0]), paddle.assign(res[1], {}[1])
                """
            )
            code = API_TEMPLATE.format(x_v, out_v, out_v)
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                res = paddle.linalg.slogdet({})
                res[0], res[1]
                """
            )
            code = API_TEMPLATE.format(x_v)

        return code


class HistcMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            input = kwargs["input"]
        else:
            input = self.paddleClass

        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            code = "paddle.assign({}({}).astype({}.dtype), output={})".format(
                self.get_paddle_api(),
                self.kwargs_to_str(kwargs),
                input,
                out_v,
            )
        else:
            code = "{}({}).astype({}.dtype)".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), input
            )

        return code


class SpecialNdtriMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            2 ** (1/2) * paddle.erfinv(2*{}-1)
            """
        )
        code = API_TEMPLATE.format(kwargs["input"])
        if "out" in kwargs and kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class AdjointMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            {} = list(range({}.ndim))
            {}[-1], {}[-2] = {}[-2], {}[-1]
            paddle.conj(paddle.transpose({}, perm={}))
            """
        )
        perm = get_unique_name("perm")
        code = API_TEMPLATE.format(
            perm, kwargs["input"], perm, perm, perm, perm, kwargs["input"], perm
        )

        return code


class TensorHMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "{}.transpose(perm=[1, 0]).conj()".format(self.paddleClass)
        return ast.parse(code).body


class TensorMtMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        paddle_class = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            {} = list(range({}.ndim))
            {}[-1], {}[-2] = {}[-2], {}[-1]
            {}.transpose(perm={})
            """
        )
        perm = get_unique_name("perm")
        code = API_TEMPLATE.format(
            perm, paddle_class, perm, perm, perm, perm, paddle_class, perm
        )
        return ast.parse(code).body


class TensorMhMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        paddle_class = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            {} = list(range({}.ndim))
            {}[-1], {}[-2] = {}[-2], {}[-1]
            {}.transpose(perm={}).conj()
            """
        )
        perm = get_unique_name("perm")
        code = API_TEMPLATE.format(
            perm, paddle_class, perm, perm, perm, perm, paddle_class, perm
        )
        return ast.parse(code).body


class SpecialXLog1pYMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            {} * paddle.log1p(paddle.to_tensor({}))
            """
        )
        code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])
        if "out" in kwargs and kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class FunctionalKLDivMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        process_reduce_and_size_average(kwargs)

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")
        log_target = kwargs.pop("log_target", False)
        API_TEMPLATE = "paddle.nn.functional.kl_div(input={}, label={}, reduction={})"
        code = API_TEMPLATE.format(
            kwargs.get("input"),
            kwargs.get("label")
            if log_target is False
            else f"paddle.exp({kwargs.get('label')})",
            kwargs.pop("reduction", '"""mean"""'),
        )
        return code


class FunctionalSmoothL1LossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        process_reduce_and_size_average(kwargs)
        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        if "beta" in kwargs:
            kwargs["delta"] = kwargs.pop("beta")
            API_TEMPLATE = "paddle.nn.functional.smooth_l1_loss({})/" + kwargs["delta"]
        else:
            API_TEMPLATE = "paddle.nn.functional.smooth_l1_loss({})"

        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class TupleAssignMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]

        for k in kwargs_change:
            if k in kwargs:
                kwargs[kwargs_change[k]] = kwargs.pop(k)

        if "out" in kwargs:
            out_v = kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                out1, out2 = {}({})
                paddle.assign(out1, {}[0]), paddle.assign(out2, {}[1])
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v, out_v
            )
            return code.strip("\n")
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
            return code.strip("\n")


class TripleAssignMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.set_paddle_default_kwargs(kwargs)
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]

        for k in kwargs_change:
            if k in kwargs:
                if kwargs_change[k]:
                    kwargs[kwargs_change[k]] = kwargs.pop(k)
                else:
                    kwargs.pop(k)

        if "out" in kwargs:
            out_v = kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                out1, out2, out3 = {}({})
                paddle.assign(out1, {}[0]), paddle.assign(out2, {}[1]), paddle.assign(out3, {}[2])
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v, out_v, out_v
            )
            return code.strip("\n")
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
            return code.strip("\n")


class RoundMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "decimals" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.round((10**{}) * {}) / (10**{})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["decimals"], kwargs["input"], kwargs["decimals"]
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.round({})
                """
            )
            code = API_TEMPLATE.format(kwargs["input"])
        if "out" in kwargs and kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class RNNMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "batch_first" in kwargs:
            batch_first = kwargs.pop("batch_first")
        else:
            batch_first = False
        kwargs["time_major"] = f"not {batch_first}"

        kwargs["direction"] = "'forward'"
        if "bidirectional" in kwargs:
            if "True" in kwargs["bidirectional"]:
                direction = "'bidirect'"
            kwargs.pop("bidirectional")

        return GenericMatcher.generate_code(self, kwargs)


class RNNBaseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "batch_first" in kwargs:
            batch_first = kwargs.pop("batch_first")
        else:
            batch_first = False
        kwargs["time_major"] = f"not {batch_first}"

        if "mode" in kwargs:
            # RNN is not supported, because RNN need activation
            if "LSTM" in kwargs["mode"] or "GRU" in kwargs["mode"]:
                pass
            else:
                return None

        direction = "'forward'"
        if "bidirectional" in kwargs:
            if "True" in kwargs["bidirectional"]:
                direction = "'bidirect'"
            kwargs.pop("bidirectional")
        kwargs["direction"] = direction

        return GenericMatcher.generate_code(self, kwargs)


class LinalgSvdvalsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs:
            out_v = kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}[1], output={})
                """
            )
            code = API_TEMPLATE.format(
                GenericMatcher.generate_code(self, kwargs), out_v
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {}[1]
                """
            )
            code = API_TEMPLATE.format(GenericMatcher.generate_code(self, kwargs))
        return code


class DiffMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "n" in kwargs and kwargs["n"] != "(1)":
            return None
        return GenericMatcher.generate_code(self, kwargs)


class Tuple2ListMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        kwargs_change = self.api_mapping["kwargs_change"]
        for k in kwargs.keys():
            if k in kwargs_change:
                if "," in kwargs[k]:
                    new_kwargs[kwargs_change[k]] = "list({})".format(kwargs[k])
                else:
                    new_kwargs[kwargs_change[k]] = kwargs[k]
            else:
                if "," in kwargs[k]:
                    new_kwargs[k] = "list({})".format(kwargs[k])
                else:
                    new_kwargs[k] = kwargs[k]

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))

        return code


class ParameterMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_args_and_kwargs(args, kwargs)
        if "requires_grad" in kwargs:
            requires_grad_v = kwargs["requires_grad"]
        else:
            requires_grad_v = "True"

        API_TEMPLATE = textwrap.dedent(
            """
            {} = paddle.create_parameter(shape={}.shape, dtype={}.numpy().dtype, default_initializer=paddle.nn.initializer.Assign({}))
            {}.stop_gradient = not {}
            {}
            """
        )
        out = get_unique_name("out")
        code = API_TEMPLATE.format(
            out,
            kwargs["data"],
            kwargs["data"],
            kwargs["data"],
            out,
            requires_grad_v,
            out,
        )
        return ast.parse(code.strip("\n")).body


class Modules_BatchNormBaseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "track_running_stats" in kwargs:
            track_running_stats = kwargs["track_running_stats"]
        else:
            track_running_stats = True
        kwargs["use_global_stats"] = track_running_stats

        if "momentum" in kwargs:
            momentum = kwargs["momentum"]
        else:
            momentum = 0.1
        kwargs["momentum"] = f"1-{momentum}"

        if "affine" not in kwargs:
            kwargs["weight_attr"] = None
            kwargs["bias_attr"] = None
        else:
            kwargs[
                "weight_attr"
            ] = f"None if ({kwargs['affine']} is None or {kwargs['affine']}) else False"
            kwargs[
                "bias_attr"
            ] = f"None if ({kwargs['affine']} is None or {kwargs['affine']}) else False"

        return GenericMatcher.generate_code(self, kwargs)


class TensorTakeMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def take(self, *args, **kwargs):
                if args:
                    return paddle.take(self, *args)
                elif kwargs:
                    return paddle.take(self, **kwargs)

            setattr(paddle.Tensor, 'take', take)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.write_aux_code()
        return "unchange"


class TensorSplitMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def split_tensor_func(self, split_size, dim=0):
                if isinstance(split_size, int):
                    return paddle.split(self, self.shape[dim]//split_size, dim)
                else:
                    return paddle.split(self, split_size, dim)

            setattr(paddle.Tensor, 'split', split_tensor_func)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        args_list = self.api_mapping.get("args_list") or []
        for k in kwargs:
            if k not in args_list:
                return "misidentify"

        if len(args) + len(kwargs) < 1 or len(args) + len(kwargs) > 2:
            return "misidentify"

        if len(args) == 1 and isinstance(args[0], ast.Constant):
            if isinstance(args[0].value, str):
                return "misidentify"

        self.write_aux_code()
        return "unchange"


class TensorRoundMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def round(self, decimals=None):
                if decimals:
                    x = paddle.abs(self)//(10**-decimals)*(10**-decimals)
                    return paddle.where(self<0, -x, x)
                return paddle.round(self)
            setattr(paddle.Tensor, 'round', round)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if args is None and kwargs is None:
            return "unchange"

        self.write_aux_code()
        return "unchange"


class TensorRound_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["input"] = self.paddleClass

        if "decimals" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign(({} * (10**{})).round_() / (10**{}), {})
                """
            )
            return API_TEMPLATE.format(
                kwargs["input"], kwargs["decimals"], kwargs["decimals"], kwargs["input"]
            )
        else:
            return "{}.round_()".format(kwargs["input"])


class NonzeroMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "as_tuple" in kwargs and kwargs["as_tuple"] != "(False)":
            return None
        return GenericMatcher.generate_code(self, kwargs)


class NormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "p" in kwargs and "nuc" in kwargs["p"]:
            return None
        return GenericMatcher.generate_code(self, kwargs)


class SortMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        change_kwargs = self.api_mapping["kwargs_change"]
        for key in change_kwargs:
            if key in kwargs:
                if change_kwargs[key]:
                    kwargs[change_kwargs[key]] = kwargs.pop(key).strip("\n")
                else:
                    kwargs.pop(key)

        if "out" not in kwargs:
            code = "paddle.sort({}), paddle.argsort({})".format(
                self.kwargs_to_str(kwargs), self.kwargs_to_str(kwargs)
            )
        else:
            out_v = kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                out1, out2 = paddle.sort({}), paddle.argsort({})
                paddle.assign(out1, {}[0]), paddle.assign(out2, {}[1])
                """
            )
            code = API_TEMPLATE.format(
                self.kwargs_to_str(kwargs), self.kwargs_to_str(kwargs), out_v, out_v
            )
        return code.strip("\n")


class WhereMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) == 1:
            return None
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class UpsampleMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size" in kwargs and "," not in kwargs["size"]:
            return None
        return GenericMatcher.generate_code(self, kwargs)


class NTupleMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import collections
            from itertools import repeat
            def _ntuple(n, name="parse"):
                def parse(x):
                    if isinstance(x, collections.abc.Iterable):
                        return tuple(x)
                    return tuple(repeat(x, n))

                parse.__name__ = name
                return parse
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.write_aux_code()
        if "x" not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle_aux._ntuple({})
                """
            )
            code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        else:
            kwargs = self.set_paddle_default_kwargs(kwargs)
            API_TEMPLATE = textwrap.dedent(
                """
                paddle_aux._ntuple({})({})
                """
            )
            code = API_TEMPLATE.format(kwargs["n"], kwargs["x"])

        return code


class Get_EnumMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def get_enum(reduction: str) -> int:
                if reduction == 'none':
                    ret = 0
                elif reduction == 'mean':
                    ret = 1
                elif reduction == 'elementwise_mean':
                    warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
                    ret = 1
                elif reduction == 'sum':
                    ret = 2
                else:
                    ret = -1
                    raise ValueError("{} is not a valid value for reduction".format(reduction))
                return ret
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.write_aux_code()
        API_TEMPLATE = textwrap.dedent(
            """
            paddle_aux.get_enum({})
            """
        )
        code = API_TEMPLATE.format(kwargs["reduction"])

        return code


class UnpoolMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["indices"] = (
            "(" + kwargs.pop("indices").strip("\n") + ").astype('int32')"
        )

        return GenericMatcher.generate_code(self, kwargs)


class SoftmaxMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" not in kwargs or "None" in kwargs["dim"]:
            return None

        return GenericMatcher.generate_code(self, kwargs)


class OptimOptimizerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "paddle.optimizer.Optimizer(parameters={}, **{})".format(
            kwargs.pop("params"), kwargs["defaults"]
        )
        return code


class OptimAdamMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "betas" in kwargs:
            kwargs["beta1"] = "{}[0]".format(kwargs["betas"])
            kwargs["beta2"] = "{}[1]".format(kwargs["betas"])
            kwargs.pop("betas")
        return GenericMatcher.generate_code(self, kwargs)


class ConstantLRMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        optim = kwargs.pop("optimizer")
        factor = 0.3333333333333333
        total_iters = 5
        if "factor" in kwargs:
            factor = kwargs.pop("factor")
        if "total_iters" in kwargs:
            total_iters = kwargs.pop("total_iters")
        kwargs["values"] = "[{}*{}.get_lr(), {}.get_lr()]".format(factor, optim, optim)
        kwargs["boundaries"] = "[{}]".format(total_iters)
        API_TEMPLATE = textwrap.dedent(
            """
            tmp_lr = {}({})
            {}.set_lr_scheduler(tmp_lr)
            tmp_lr
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs), optim
        )
        return code


class LrSchedulerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        optim = kwargs.pop("optimizer")
        if self.get_paddle_api() not in [
            "paddle.optimizer.lr.CyclicLR",
        ]:
            kwargs["learning_rate"] = optim + ".get_lr()"
        API_TEMPLATE = textwrap.dedent(
            """
            tmp_lr = {}({})
            {}.set_lr_scheduler(tmp_lr)
            tmp_lr
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs), optim
        )
        return code


class FunctionalSoftmaxMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" not in kwargs or "None" in kwargs["dim"]:
            return None

        return GenericMatcher.generate_code(self, kwargs)


class FunctionalLinearMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["weight"] = kwargs["weight"].strip("\n") + ".T"

        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input").strip("\n")

        return "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class FunctionalBilinearMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "bias" in kwargs:
            kwargs["bias"] = kwargs["bias"].strip("\n") + ".unsqueeze(0)"

        kwargs["x1"] = kwargs.pop("input1").strip("\n")
        kwargs["x2"] = kwargs.pop("input2").strip("\n")

        return "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class FunctionalOneHotMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = "{}.max().item() + 1".format(kwargs["input"])

        kwargs["x"] = kwargs.pop("input").strip("\n")

        return "{}({}).astype('int64')".format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs)
        )


class SizeAverageMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        process_reduce_and_size_average(kwargs)
        return GenericMatcher.generate_code(self, kwargs)


class CudaStreamMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "priority" in kwargs:
            kwargs["priority"] = "{}+2".format(kwargs["priority"])

        if "device" in kwargs:
            if "cuda" in kwargs["device"]:
                import re

                device_list = re.findall(r"\d+", kwargs["device"])
                if len(device_list) > 0:
                    kwargs["device"] = device_list[0]
                else:
                    kwargs["device"] = None

        return GenericMatcher.generate_code(self, kwargs)


class CudaNvtxRangePushMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}({})".format(self.get_paddle_api(), kwargs["msg"])
        return code


class Attribute2Func(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "{}()".format(self.paddle_api)
        return ast.parse(code).body


class LuMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        out_v = kwargs.pop("out") if "out" in kwargs else None

        if out_v:
            get_infos_v = "get_infos" in kwargs and kwargs["get_infos"] != "(False)"
            new_kwargs = {}
            new_kwargs["x"] = kwargs.pop("A")
            new_kwargs.update(kwargs)

            if get_infos_v:
                API_TEMPLATE = textwrap.dedent(
                    """
                    tmp_lu, tmp_p, tmp_info = {}({})
                    paddle.assign(tmp_lu, {}[0]), paddle.assign(tmp_p, {}[1]), paddle.assign(tmp_info, {}[2])
                    """
                )
                code = API_TEMPLATE.format(
                    self.get_paddle_api(),
                    self.kwargs_to_str(new_kwargs),
                    out_v,
                    out_v,
                    out_v,
                )
            else:
                API_TEMPLATE = textwrap.dedent(
                    """
                    tmp_lu, tmp_p = {}({})
                    paddle.assign(tmp_lu, {}[0]), paddle.assign(tmp_p, {}[1])
                    """
                )
                code = API_TEMPLATE.format(
                    self.get_paddle_api(), self.kwargs_to_str(new_kwargs), out_v, out_v
                )

            return code

        return GenericMatcher.generate_code(self, kwargs)


class LinalgLuMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        out_v = kwargs.pop("out") if "out" in kwargs else None

        new_kwargs = {}
        new_kwargs["x"] = kwargs.pop("A")
        new_kwargs.update(kwargs)
        if out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                tmp_lu, tmp_p = paddle.linalg.lu({})
                P, L, U = paddle.linalg.lu_unpack(tmp_lu, tmp_p)
                paddle.assign(P, {}[0]), paddle.assign(L, {}[1]), paddle.assign(U, {}[2])
                """
            )
            code = API_TEMPLATE.format(
                self.kwargs_to_str(new_kwargs),
                out_v,
                out_v,
                out_v,
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                tmp_lu, tmp_p = paddle.linalg.lu({})
                paddle.linalg.lu_unpack(tmp_lu, tmp_p)
                """
            )
            code = API_TEMPLATE.format(
                self.kwargs_to_str(new_kwargs),
            )
        return code


class LinalgSolveTriangularMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        if "left" in kwargs:
            new_kwargs["transpose"] = f"not {kwargs.pop('left')}"
        new_kwargs.update(kwargs)
        return GenericMatcher.generate_code(self, new_kwargs)


class QrMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        some_v = kwargs.pop("some") if "some" in kwargs else None
        out_v = kwargs.pop("out") if "out" in kwargs else None

        if some_v:
            kwargs["mode"] = "'complete'" if some_v != "(False)" else "'reduced'"

        if out_v:
            kwargs["x"] = kwargs.pop("input")
            API_TEMPLATE = textwrap.dedent(
                """
                tmp_q, tmp_r = {}({})
                paddle.assign(tmp_q, {}[0]), paddle.assign(tmp_r, {}[1])
                """
            )

            code = API_TEMPLATE.format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v, out_v
            )
            return code

        return GenericMatcher.generate_code(self, kwargs)


class SvdMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "compute_uv" in kwargs:
            return None

        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")
        else:
            kwargs["x"] = self.paddleClass

        if "some" in kwargs:
            kwargs["full_matrices"] = "not " + kwargs.pop("some").strip("()")

        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            API_TEMPLATE = textwrap.dedent(
                """
                tmp_u, tmp_s, tmp_v = {}({})
                paddle.assign(tmp_u, {}[0]), paddle.assign(tmp_s, {}[1]), paddle.assign(tmp_v.conj().t(), {}[2])
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(),
                self.kwargs_to_str(kwargs),
                out_v,
                out_v,
                out_v,
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                tmp_u, tmp_s, tmp_v = {}({})
                tmp_u, tmp_s, tmp_v.conj().t()
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(),
                self.kwargs_to_str(kwargs),
            )

        return code


class ModuleGetSubMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = 'getattr({}, "{}")'.format(
            self.paddleClass,
            kwargs["target"].strip('"'),
        )
        return code


class TensorIsSignedMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        paddle_class = self.paddleClass
        code = "{}.dtype not in [paddle.uint8]".format(paddle_class)
        return ast.parse(code.strip("\n")).body


class TensorToBoolMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim").strip("\n")

        paddle_api = self.get_paddle_api()
        paddle_api_name = paddle_api[paddle_api.rfind(".") :]
        code = "{}({})".format(
            self.paddleClass + ".astype('bool')" + paddle_api_name,
            self.kwargs_to_str(kwargs),
        )
        return code


class TensorDatasetMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        tensors_v = "[{}".format(new_args[0])
        for arg in new_args[1:]:
            tensors_v += ", {}".format(arg)
        tensors_v += "]"
        code = "{}({})".format(self.get_paddle_api(), tensors_v)
        node = ast.parse(code.strip("\n")).body
        return node


class TensorMaxMinMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        new_kwargs = self.parse_kwargs(kwargs)
        if new_kwargs is None:
            return None

        call_maximinimum = False
        if len(args) > 0 and not isinstance(args[0], ast.Num):
            call_maximinimum = True

        if "other" in new_kwargs:
            call_maximinimum = True

        # the case of two Tensor
        if call_maximinimum:
            return GenericMatcher(
                self.transformer, self.torch_api, self.api_mapping, self.logger
            ).get_paddle_class_nodes(func, args, kwargs)

        # the case of one Tensor
        args_list = ["dim", "keepdim"]

        # parse args to kwargs
        for i in range(len(args)):
            new_kwargs[args_list[i]] = astor.to_source(args[i]).strip("\n")
        for node in kwargs:
            new_kwargs[node.arg] = astor.to_source(node.value).strip("\n")

        # change kwargs' name
        if "dim" in new_kwargs:
            new_kwargs["axis"] = new_kwargs.pop("dim")

        if "min" in self.torch_api:
            paddle_api, paddle_api_arg = (
                self.paddleClass + ".min",
                self.paddleClass + ".argmin",
            )
        else:
            paddle_api, paddle_api_arg = (
                self.paddleClass + ".max",
                self.paddleClass + ".argmax",
            )

        if "axis" in new_kwargs:
            return ast.parse(
                "{}({}), {}({})".format(
                    paddle_api,
                    self.kwargs_to_str(new_kwargs),
                    paddle_api_arg,
                    self.kwargs_to_str(new_kwargs),
                )
            ).body
        else:
            return ast.parse(
                "{}({})".format(paddle_api, self.kwargs_to_str(new_kwargs))
            ).body


class Func2Attribute(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}".format(self.get_paddle_api())
        return code


class SetUpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        is_cpp_extension = False
        if "cmdclass" in kwargs:
            if "paddle.utils.cpp_extension.BuildExtension" in kwargs["cmdclass"]:
                is_cpp_extension = True

        if not is_cpp_extension:
            return "misidentify"

        kwargs.pop("cmdclass")
        global CPP_EXTENSION_LIST
        CPP_EXTENSION_LIST.append(kwargs["name"].strip('"'))
        return ast.parse(
            "paddle.utils.cpp_extension.setup({})".format(self.kwargs_to_str(kwargs))
        )
