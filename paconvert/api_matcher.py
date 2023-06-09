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

import ast
import textwrap

import astor

from paconvert.base import BaseMatcher
from paconvert.utils import get_unique_name


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
                if k in [
                    "layout",
                    "device",
                    "memory_format",
                    "inplace",
                    "generator",
                    "non_blocking",
                ]:
                    new_kwargs.pop(k)
                    continue
                if k == "dtype":
                    dtype_v = new_kwargs.pop("dtype")

        new_kwargs = self.set_paddle_default_kwargs(new_kwargs)

        pin_memory_v = False
        if "pin_memory" in new_kwargs:
            pin_memory_v = eval(new_kwargs.pop("pin_memory"))

        stop_gradient_v = None
        if "requires_grad" in new_kwargs:
            stop_gradient_v = "not " + new_kwargs.pop("requires_grad").strip("()")

        out_v = new_kwargs.pop("out") if "out" in new_kwargs else None

        res = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))

        if dtype_v:
            res += ".astype({})".format(dtype_v)

        if pin_memory_v:
            res += ".pin_memory()"

        if stop_gradient_v and out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                x = {}
                x.stop_gradient = {}
                paddle.assign(x, output={})
                """
            )
            code = API_TEMPLATE.format(res, stop_gradient_v, out_v)
        elif stop_gradient_v and not out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(out, res, out, stop_gradient_v, out)
        elif not stop_gradient_v and out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}, output={})
                """
            )
            code = API_TEMPLATE.format(res, out_v)
        else:
            code = "{}".format(res)

        return code

    def get_paddle_class_attribute_nodes(self, node):
        node.attr = ast.parse(self.get_paddle_api()).body[0].value.attr
        return node


class DeleteMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        return "delete"

    def get_paddle_api(self):
        return "delete"


class TensorDeleteMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        return "delete"

    def get_paddle_class_attribute_nodes(self, node):
        return "delete"


class TensorUnchangeMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        return "unchange"

    def get_paddle_class_attribute_nodes(self, node):
        return "unchange"


class IdentityMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        code = "{}({})".format(
            self.get_paddle_api(), self.args_and_kwargs_to_str(new_args, new_kwargs)
        )
        return ast.parse(code).body


class LayerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "device" in kwargs:
            del kwargs["device"]
        if "dtype" in kwargs:
            del kwargs["dtype"]
        if "bias" in kwargs:
            kwargs["bias_attr"] = kwargs.pop("bias")
        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class TorchAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "alpha" in kwargs:
            code = "paddle.add(x={}, y=paddle.to_tensor({})*{})".format(
                kwargs["input"], kwargs["alpha"], kwargs["other"]
            )
        else:
            code = "paddle.add(x={}, y=paddle.to_tensor({}))".format(
                kwargs["input"], kwargs["other"]
            )

        if "out" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}, output={})
                """
            )
            code = API_TEMPLATE.format(code, kwargs["out"])

        return code


class TensorAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "alpha" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.Tensor.add(y={}*{})
                """
            )
            code = API_TEMPLATE.format(kwargs["alpha"], kwargs["other"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.Tensor.add(y={})
                """
            )
            code = API_TEMPLATE.format(kwargs["other"])
        return code


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

        pin_memory_v = False
        if "pin_memory" in kwargs:
            pin_memory_v = eval(kwargs.pop("pin_memory"))

        stop_gradient_v = None
        if "requires_grad" in kwargs:
            stop_gradient_v = "not " + kwargs.pop("requires_grad").strip("()")

        out_v = kwargs.pop("out") if "out" in kwargs else None

        res = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        if pin_memory_v:
            res += ".pin_memory()"

        if stop_gradient_v and out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                x = {}
                x.stop_gradient = {}
                paddle.assign(x, output={})
                """
            )
            code = API_TEMPLATE.format(res, stop_gradient_v, out_v)
        elif stop_gradient_v and not out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(out, res, out, stop_gradient_v, out)
        elif not stop_gradient_v and out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}, output={})
                """
            )
            code = API_TEMPLATE.format(res, out_v)
            code = API_TEMPLATE.format(res, out_v)
        else:
            code = "{}".format(res)

        return ast.parse(code).body


class DeviceMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        if len(new_args) == 1:
            code = f'str({new_args[0]}.replace("cuda", "gpu"))'

        if len(args) == 2:
            code = f'":".join([{astor.to_source(args[0])}.replace("cuda", "gpu"),str({astor.to_source(args[1])})])'
        return ast.parse(code).body


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
        # call maximum usage, convert
        call_maximinimum = False
        if len(args) > 1 and isinstance(args[1], (ast.Name, ast.Subscript)):
            call_maximinimum = True

        new_kwargs = self.parse_kwargs(kwargs)
        if "other" in new_kwargs:
            call_maximinimum = True

        if call_maximinimum:
            return GenericMatcher(
                self.torch_api, self.api_mapping, self.logger
            ).get_paddle_nodes(args, kwargs)

        # return (values, indices) and paddle not implement, not convert
        if len(args) > 1 and isinstance(args[1], ast.Num):
            return None
        if "dim" in new_kwargs:
            return None

        # only return values, not return indices, convert
        paddle_api = self.torch_api.replace("torch", "paddle")
        if len(args) == 1:
            x_v = astor.to_source(args[0]).strip("\n")
            return ast.parse("{}(x={})".format(paddle_api, x_v)).body

        if "input" in new_kwargs:
            x_v = new_kwargs["input"]
            return ast.parse("{}(x={})".format(paddle_api, x_v)).body

        return None


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
                if "torch.IntTensor" == self.torch_api:
                    code = "paddle.to_tensor(data={}, dtype='int32')".format(data)
                elif "torch.LongTensor" == self.torch_api:
                    code = "paddle.to_tensor(data={}, dtype='int64')".format(data)
                elif "torch.FloatTensor" == self.torch_api:
                    code = "paddle.to_tensor(data={}, dtype='float32')".format(data)
                else:
                    if not isinstance(args[0], ast.Name):
                        code = "paddle.to_tensor(data={}, dtype='float32')".format(data)
                    else:
                        code = "paddle.to_tensor(data={})".format(data)
                node = ast.parse(code.strip("\n")).body
                return node
            shape = str(shape).replace("'", "")

        if "torch.IntTensor" == self.torch_api:
            code = "paddle.empty(shape={}, dtype='int32')".format(shape)
        elif "torch.LongTensor" == self.torch_api:
            code = "paddle.empty(shape={}, dtype='int64')".format(shape)
        elif "torch.FloatTensor" == self.torch_api:
            code = "paddle.empty(shape={}, dtype='float32')".format(shape)
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


class TensorTransposeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        # may be ndarray.transpose([list]) / ndarray.transpose(list)
        if len(kwargs) != 2:
            return "NonTorchClass"

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
        if "dims" in kwargs:
            kwargs = {"perm": kwargs.pop("dims"), **kwargs}
        else:
            kwargs = {"perm": str(perm_list).replace("'", ""), **kwargs}

        code = "{}.transpose({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorRepeatMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)

        if "axis" in kwargs:
            return "NonTorchClass"

        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            repeat_list = self.parse_args(args)[0]
        elif len(args) >= 1:
            repeat_list = self.parse_args(args)

        if "repeats" in kwargs:
            kwargs = {"repeat_times": kwargs.pop("repeats"), **kwargs}
        else:
            kwargs = {"repeat_times": str(repeat_list).replace("'", ""), **kwargs}

        code = "{}.tile({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
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
        code = "{}.astype(dtype={}.dtype)".format(self.paddleClass, kwargs["tensor"])
        return code


class TensorNew_Matcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if None in kwargs:
            kwargs.pop(None)
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
        if "pin_memory" in kwargs:
            pin_memory_v = eval(kwargs.pop("pin_memory"))

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
        if "pin_memory" in kwargs:
            pin_memory_v = eval(kwargs.pop("pin_memory"))

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

        if "pin_memory" in kwargs:
            if eval(kwargs["pin_memory"]):
                kwargs["place"] = "paddle.CUDAPinnedPlace()"
            kwargs.pop("pin_memory")

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

        if "pin_memory" in kwargs:
            if eval(kwargs["pin_memory"]):
                kwargs["place"] = "paddle.CUDAPinnedPlace()"
            kwargs.pop("pin_memory")

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code.strip("\n")


class TensorNormal_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["shape"] = "x.shape"
        API_TEMPLATE = textwrap.dedent(
            """
            x = {}
            paddle.assign(paddle.normal({}).astype(x.dtype), x)
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return code.strip("\n")


class CrossEntropyLossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.nn.CrossEntropyLoss({})
            """
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class CudaIsAvailableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}() >= 1".format(self.get_paddle_api().strip("\n"))
        return code


class FunctionInterpolateMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = {}
        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
        new_kwargs = {}
        for k in list(kwargs.keys()):
            if k in kwargs_change:
                if kwargs_change[k]:
                    new_kwargs[kwargs_change[k]] = kwargs.pop(k)
            else:
                # TODO: should handle these args specially
                if k in ["recompute_scale_factor", "antialias"]:
                    kwargs.pop(k)
                    continue

                # TODO: kwargs_change -> kwargs_mapping
                # not mapping in kwargs in there is not in kwargs_mapping
                new_kwargs[k] = kwargs[k]

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(new_kwargs))
        return code.strip("\n")


class LayerNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "eps" not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs["eps"]

        if "elementwise_affine" in kwargs and "False" in kwargs["elementwise_affine"]:
            API_TEMPLACE = textwrap.dedent(
                """
                paddle.nn.LayerNorm(normalized_shape={},
                                    epsilon={},
                                    weight_attr=False,
                                    bias_attr=False)
                """
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                """
                paddle.nn.LayerNorm(normalized_shape={},
                                    epsilon={},
                                    weight_attr=None,
                                    bias_attr=None)
                """
            )
        code = API_TEMPLACE.format(kwargs["normalized_shape"], epsilon)
        return code


class GroupNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "eps" not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs["eps"]

        if "affine" in kwargs and "False" in kwargs["affine"]:
            API_TEMPLACE = textwrap.dedent(
                """
                paddle.nn.GroupNorm(num_groups={},
                                    num_channels={},
                                    epsilon={},
                                    weight_attr=False,
                                    bias_attr=False)
                """
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                """
                paddle.nn.GroupNorm(num_groups={},
                                    num_channels={},
                                    epsilon={},
                                    weight_attr=None,
                                    bias_attr=None)
                """
            )
        code = API_TEMPLACE.format(
            kwargs["num_groups"], kwargs["num_channels"], epsilon
        )
        return code


class BatchNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "eps" not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs["eps"]

        if "track_running_stats" in kwargs:
            track_running_stats = kwargs["track_running_stats"]
        else:
            track_running_stats = True

        if "momentum" in kwargs:
            momentum = kwargs["momentum"]
        else:
            momentum = 0.1

        if "affine" in kwargs and "False" in kwargs["affine"]:
            API_TEMPLACE = textwrap.dedent(
                """
                {}(num_features={},
                    momentum=1-{},
                    epsilon={},
                    weight_attr=False,
                    bias_attr=False,
                    use_global_stats={})
                """
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                """
                {}(num_features={},
                    momentum=1-{},
                    epsilon={},
                    weight_attr=None,
                    bias_attr=None,
                    use_global_stats={})
                """
            )
        code = API_TEMPLACE.format(
            self.get_paddle_api(),
            kwargs["num_features"],
            momentum,
            epsilon,
            track_running_stats,
        )
        return code


class MaxPool2DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dilation" in kwargs:
            if kwargs["dilation"] != "(1)":
                return None
            else:
                kwargs.pop("dilation")

        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
            for key in list(kwargs_change.keys()):
                if key in kwargs:
                    kwargs[kwargs_change[key]] = kwargs[key]
                    kwargs.pop(key)

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.nn.MaxPool2D({})
            """
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class SplitMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" in kwargs:
            axis = kwargs["dim"]
        else:
            axis = 0

        if "[" in kwargs["split_size_or_sections"]:
            API_TEMPLACE = textwrap.dedent(
                """
                paddle.split(x={}, num_or_sections={}, axis={})
                """
            )
            code = API_TEMPLACE.format(
                kwargs["tensor"], kwargs["split_size_or_sections"], axis
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                """
                paddle.split(x={}, num_or_sections={}.shape[{}]//{}, axis={})
                """
            )
            code = API_TEMPLACE.format(
                kwargs["tensor"],
                kwargs["tensor"],
                axis,
                kwargs["split_size_or_sections"],
                axis,
            )
        return code


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
        API_TEMPLACE = textwrap.dedent(
            """
            {} = paddle.arange(start={}, end={}+1 if ({} - {}) % {} == 0 else {}, step={}, dtype={})
            {}.stop_gradient = not {}
            {}
            """
        )
        code = API_TEMPLACE.format(
            out,
            start,
            kwargs["end"],
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
        if "indexing" in new_kwargs:
            if "ij" not in new_kwargs["indexing"]:
                return None
        code = "{}({})".format(self.get_paddle_api(), self.args_to_str(new_args))
        return ast.parse(code).body


class TensorIsContiguousMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        code = "True"
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
        API_TEMPLACE = textwrap.dedent(
            """
            paddle.assign({}, output={})
            """
        )
        code = API_TEMPLACE.format(args[0], self.paddleClass)
        return ast.parse(code).body


class TensorMaskedFillMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)
        API_TEMPLACE = textwrap.dedent(
            """
            paddle.where({}, {}, {})
            """
        )
        code = API_TEMPLACE.format(kwargs["mask"], self.paddleClass, kwargs["value"])
        return ast.parse(code).body


class TensorUniqueMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if "sorted" in kwargs:
            if "False" in kwargs["sorted"]:
                return None
            else:
                kwargs.pop("sorted")

        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
            for key in list(kwargs_change.keys()):
                if key in kwargs:
                    kwargs[kwargs_change[key]] = kwargs[key]
                    kwargs.pop(key)

        API_TEMPLACE = textwrap.dedent(
            """
            {}.unique({})
            """
        )
        code = API_TEMPLACE.format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorExpandMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
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
            axis = kwargs["dim"]
        else:
            return None

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.nn.functional.softmax({}, axis={})
            """
        )
        code = API_TEMPLACE.format(self.paddleClass, axis)
        return ast.parse(code).body


class TensorRequiresGradMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        if "requires_grad" in kwargs:
            requires_grad_v = kwargs["requires_grad"]
        else:
            requires_grad_v = "True"

        API_TEMPLACE = textwrap.dedent(
            """
            {} = {}
            {}.stop_gradient = not {}
            {}
            """
        )
        out = get_unique_name("out")
        code = API_TEMPLACE.format(out, self.paddleClass, out, requires_grad_v, out)
        return ast.parse(code.strip("\n")).body


class FunctionalL1LossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.l1_loss({})
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class FunctionalBinaryCrossEntropyWithLogitsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        if "input" in kwargs:
            kwargs["logit"] = kwargs.pop("input")

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.binary_cross_entropy_with_logits({})
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class FunctionalMaxPool2DMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dilation" in kwargs:
            if kwargs["dilation"] != "(1)":
                return None
            else:
                kwargs.pop("dilation")

        if "kwargs_change" in self.api_mapping:
            kwargs_change = self.api_mapping["kwargs_change"]
            for key in list(kwargs_change.keys()):
                if key in kwargs:
                    kwargs[kwargs_change[key]] = kwargs[key]
                    kwargs.pop(key)

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.nn.functional.max_pool2d({})
            """
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


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

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.load(path={})
            """
        )
        code = API_TEMPLACE.format(kwargs["f"])
        return code


class SaveMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "pickle_module" in kwargs:
            kwargs.pop("pickle_module")

        if "_use_new_zipfile_serialization" in kwargs:
            kwargs.pop("_use_new_zipfile_serialization")

        if "pickle_protocol" in kwargs:
            protocol = kwargs["pickle_protocol"]
        else:
            protocol = 4

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.save(obj={}, path={}, protocol={})
            """
        )
        code = API_TEMPLACE.format(kwargs["obj"], kwargs["f"], protocol)
        return code


class SeedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.get_cuda_rng_state()[0].current_seed()
            """
        )
        return API_TEMPLATE


class SetPrintOptionsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "profile" in kwargs and kwargs["profile"] is not None:
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
        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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
        if "out" in kwargs and kwargs["out"] is not None:
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
        if "out" in kwargs and kwargs["out"] is not None:
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
        if "out" in kwargs and kwargs["out"] is not None:
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


class InstanceNormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "eps" not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs["eps"]

        if "momentum" in kwargs:
            momentum = kwargs["momentum"]
        else:
            momentum = 0.1

        if "affine" in kwargs and "True" in kwargs["affine"]:
            API_TEMPLACE = textwrap.dedent(
                """
                {}(num_features={},
                    momentum=1-{},
                    epsilon={},
                    weight_attr=None,
                    bias_attr=None)
                """
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                """
                {}(num_features={},
                    momentum=1-{},
                    epsilon={},
                    weight_attr=False,
                    bias_attr=False)
                """
            )

        code = API_TEMPLACE.format(
            self.get_paddle_api(), kwargs["num_features"], momentum, epsilon
        )
        return code


class BCEWithLogitsLossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        API_TEMPLACE = textwrap.dedent(
            """
            paddle.nn.BCEWithLogitsLoss({})
            """
        )
        code = API_TEMPLACE.format(self.kwargs_to_str(kwargs))
        return code


class GeneratorMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if not kwargs:
            code = "paddle.fluid.core.default_cpu_generator()"
        elif "device" in kwargs:
            if kwargs["device"] == '"""cuda"""':
                code = textwrap.dedent(
                    """
                    device = paddle.device.get_device()
                    paddle.fluid.core.default_cuda_generator(int(device[-1]))
                    """
                )
            elif kwargs["device"] == '"""mps"""':
                # paddle not suppor mps, but support xpu
                return None

            else:
                code = "paddle.fluid.core.default_cpu_generator()"

        return code


class TorchUtilDataBatchSampler(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.io.BatchSampler(sampler = {} if isinstance({}, paddle.io.Sampler) else paddle.io.SequenceSampler({}), batch_size = {}, drop_last = {})
             """
        )

        code = API_TEMPLATE.format(
            kwargs["sampler"],
            kwargs["sampler"],
            kwargs["sampler"],
            kwargs["batch_size"],
            kwargs["drop_last"],
        )

        return code


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
        self.parse_func(func)
        self.write_aux_code()
        new_kwargs = self.parse_args_and_kwargs(args, kwargs)
        API_TEMPLACE = textwrap.dedent(
            """
            import sys
            sys.path.append('{}')
            import paddle_aux
            {}.to({})
            """
        )
        code = API_TEMPLACE.format(
            self.get_aux_dir(),
            self.paddleClass,
            self.kwargs_to_str(new_kwargs),
        )
        return ast.parse(code).body


class TensorRequires_GradMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "not {}.stop_gradient".format(self.paddleClass)
        return ast.parse(code).body[0].value


class AllMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["input"] = kwargs["input"] + ".astype(dtype='bool')"
        code = GenericMatcher.generate_code(self, kwargs)
        return code


class ArangeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "end" not in kwargs:
            kwargs["end"] = kwargs.pop("start")
        if "dtype" not in kwargs and "." in kwargs["end"]:
            kwargs["dtype"] = "'float32'"
        code = GenericMatcher.generate_code(self, kwargs)
        return code


class ErfCMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] is not None:
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


class Exp2Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] is not None:
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


class FModMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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
        if "input" in kwargs:
            kwargs["input"] = kwargs.pop("input").strip("\n") + ".astype('float32')"
        else:
            kwargs["input"] = self.paddleClass

        if "other" in kwargs:
            kwargs["other"] = kwargs.pop("other").strip("\n") + ".astype('float32')"

        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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
        if "correction" not in kwargs and "unbiased" not in kwargs:
            kwargs["correction"] = 1

        if "unbiased" in kwargs:
            kwargs["correction"] = kwargs["unbiased"]

        if "keepdim" not in kwargs:
            kwargs["keepdim"] = False

        if "dim" not in kwargs:
            kwargs["dim"] = None

        if "out" in kwargs and kwargs["out"] is not None:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign((paddle.std({}, axis={}, unbiased={}, keepdim={}), paddle.mean({}, axis={}, keepdim={})), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["correction"],
                kwargs["keepdim"],
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
                kwargs["correction"],
                kwargs["keepdim"],
                kwargs["input"],
                kwargs["dim"],
                kwargs["keepdim"],
            )

        return code


class VarMeanMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "correction" not in kwargs and "unbiased" not in kwargs:
            kwargs["correction"] = 1

        if "unbiased" in kwargs:
            kwargs["correction"] = kwargs["unbiased"]

        if "keepdim" not in kwargs:
            kwargs["keepdim"] = False

        if "dim" not in kwargs:
            kwargs["dim"] = None

        if "out" in kwargs and kwargs["out"] is not None:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign((paddle.var({}, axis={}, unbiased={}, keepdim={}), paddle.mean({}, axis={}, keepdim={})), output={})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"],
                kwargs["dim"],
                kwargs["correction"],
                kwargs["keepdim"],
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
                kwargs["correction"],
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

        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
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
        if "dim" not in kwargs or kwargs["dim"] is None:
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
        if "input" in kwargs:
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
        if "input" in kwargs:
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


class FUpsampleNearestMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.upsample({}, mode='nearest')
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))

        return code


class FUpsampleBilinearMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.upsample({}, mode='bilinear', align_corners=True)
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))

        return code


class TensorBernoulli_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "p" not in kwargs:
            kwargs["p"] = 0.5
        API_TEMPLATE = textwrap.dedent(
            """
            {} = paddle.to_tensor([{}], dtype=paddle.float32) if not isinstance({}, paddle.Tensor) else {}
            paddle.assign(paddle.bernoulli(paddle.broadcast_to({}, {}.shape)), {})
            """
        )
        bern = get_unique_name("bern")
        code = API_TEMPLATE.format(
            bern,
            kwargs["p"],
            kwargs["p"],
            kwargs["p"],
            bern,
            self.paddleClass,
            self.paddleClass,
        )

        return code


class MSortMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] is not None:
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


class ExpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = "(" + kwargs.pop("input").strip("\n") + ").astype('float32')"

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class TensorSVDMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "some" in kwargs:
            kwargs["full_matrices"] = not kwargs.pop("some")
        else:
            kwargs["full_matrices"] = False

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.linalg.svd({}, full_matrices={})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["full_matrices"])

        return code


class TensorVDotMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.sum({} * {})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["other"])

        return code


class UnflattenMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.reshape({}, shape=({}.shape[:{}] + list({}) + ({}.shape[{}+1:] if {} !=-1 else [])))
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"],
            kwargs["input"],
            kwargs["dim"],
            kwargs["sizes"],
            kwargs["input"],
            kwargs["dim"],
            kwargs["dim"],
        )

        return code


class NumelMatcher(BaseMatcher):
    def generate_code(self, kwargs):
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

        if "out" in kwargs and kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])
        return code


class LogicalMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")

        if "other" in kwargs:
            kwargs["y"] = kwargs.pop("other")

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = (
                "paddle.assign({}(x={}, y=({}).astype(({}).dtype)), output={})".format(
                    self.get_paddle_api(), kwargs["x"], kwargs["y"], kwargs["x"], out_v
                )
            )
        else:
            code = "{}(x={}, y=({}).astype(({}).dtype))".format(
                self.get_paddle_api(), kwargs["x"], kwargs["y"], kwargs["x"]
            )

        return code


class AMinMaxMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "dim" not in kwargs:
            kwargs["dim"] = None

        if "keepdim" not in kwargs:
            kwargs["keepdim"] = False

        if "out" in kwargs and kwargs["out"] is not None:
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


class MulMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({} * {}, output={})".format(
                kwargs["input"], kwargs["other"], out_v
            )
        else:
            code = "{} * {}".format(kwargs["input"], kwargs["other"])

        return code


class TrueDivideMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({} / {}, output={})".format(
                kwargs["input"], kwargs["other"], out_v
            )
        else:
            code = "{} / {}".format(kwargs["input"], kwargs["other"])
        return code


class TensorDiagMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "diagonal" not in kwargs:
            kwargs["diagonal"] = 0

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.diag({}, offset={})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["diagonal"])

        return code


class DivMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.divide({}, {})
            """
        )
        code = API_TEMPLATE.format(kwargs["input"], kwargs["other"])

        if "rounding_mode" in kwargs and kwargs["rounding_mode"] is not None:
            if "trunc" in kwargs["rounding_mode"]:
                code = "paddle.trunc({})".format(code)
            elif "floor" in kwargs["rounding_mode"]:
                code = "paddle.floor({})".format(code)

        if "out" in kwargs and kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class LogsumexpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["input"] = kwargs["input"] + ".astype(dtype='float32')"
        code = GenericMatcher.generate_code(self, kwargs)
        return code


class AllcloseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = GenericMatcher.generate_code(self, kwargs)
        code = code.strip("\n") + ".item()"
        return code


class Num2TensorBinaryMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["x"] = kwargs.pop("input").strip("\n")
        kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("other").strip("\n"))
        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class SubMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1

        API_TEMPLATE = textwrap.dedent(
            """
            {} - {} * {}
            """
        )
        code = API_TEMPLATE.format(kwargs["input"], kwargs["alpha"], kwargs["other"])

        if "out" in kwargs and kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class Chain_MatmulMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)

        code = "{}".format(new_args[0])
        for arg in new_args[1:]:
            code = code + " @ {}".format(arg)
        if "out" in new_kwargs and new_kwargs["out"] is not None:
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

        if "out" in kwargs and kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class TensorHistcMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "bins" not in kwargs:
            kwargs["bins"] = 100
        if "min" not in kwargs:
            kwargs["min"] = 0
        if "max" not in kwargs:
            kwargs["max"] = 0

        API_TEMPLATE = textwrap.dedent(
            """
            {}.histogram(bins={}, min={}, max={}).astype({}.dtype)
            """
        )
        code = API_TEMPLATE.format(
            self.paddleClass,
            kwargs["bins"],
            kwargs["min"],
            kwargs["max"],
            self.paddleClass,
        )

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
                    return paddle.reshape(self, **kwargs)

            setattr(paddle.Tensor, 'reshape', reshape)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
            return "unchange"

        if len(kwargs) == 1 and "shape" in kwargs:
            return "unchange"

        self.write_aux_code()
        self.parse_func(func)
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            import sys
            sys.path.append('{}')
            import paddle_aux
            {}.reshape({})
            """
        )
        code = API_TEMPLATE.format(
            self.get_aux_dir(),
            self.paddleClass,
            self.args_and_kwargs_to_str(new_args, new_kwargs),
        )
        return ast.parse(code).body


class TensorIstftMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.signal.istft({}, {})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, self.kwargs_to_str(kwargs))

        return code


class TensorPinverseMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.linalg.pinv({})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass)

        return code


class TensorReshape_asMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.reshape({}, {}.shape)
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["other"])

        return code


class SelectMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.index_select({}, index=paddle.to_tensor([{}]), axis={}).squeeze({})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"], kwargs["index"], kwargs["dim"], kwargs["dim"]
        )

        return code


class SincMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "out" in kwargs and kwargs["out"] is not None:
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


class CumprodMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        kwargs["x"] = kwargs.pop("input").strip("\n")

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class CumsumMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        kwargs["x"] = kwargs.pop("input").strip("\n")

        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim").strip("\n")

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class SLogDetMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        out_v = kwargs.pop("out") if "out" in kwargs else None

        if "input" in kwargs:
            kwargs["A"] = kwargs.pop("input")

        if out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                res = paddle.linalg.slogdet({})
                paddle.assign(res[0], {}[0]), paddle.assign(res[1], {}[1])
                """
            )
            code = API_TEMPLATE.format(kwargs["A"], out_v, out_v)
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                res = paddle.linalg.slogdet({})
                res[0], res[1]
                """
            )
            code = API_TEMPLATE.format(kwargs["A"])

        return code


class HistcMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "out" in kwargs and kwargs["out"] is not None:
            out_v = kwargs.pop("out").strip("\n")
            code = "paddle.assign({}({}).astype('float32'), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({}).astype('float32')".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs)
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
        if "out" in kwargs and kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class TensorAtan2Matcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_args_and_kwargs(args, kwargs)

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.atan2({}, {})
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, kwargs["other"])

        return ast.parse(code).body


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


class SpecialXLog1pYMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        API_TEMPLATE = textwrap.dedent(
            """
            {} * paddle.log1p({} if isinstance({}, paddle.Tensor) else paddle.to_tensor([{}]))
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"], kwargs["other"], kwargs["other"], kwargs["other"]
        )
        if "out" in kwargs and kwargs["out"] is not None:
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class CovMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "correction" in kwargs:
            if kwargs["correction"].strip("()") == "1":
                kwargs["ddof"] = True
            elif kwargs["correction"].strip("()") == "0":
                kwargs["ddof"] = False
            else:
                return None
        else:
            kwargs["ddof"] = True

        if "fweights" not in kwargs:
            kwargs["fweights"] = None

        if "aweights" not in kwargs:
            kwargs["aweights"] = None

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.linalg.cov({}, ddof={}, fweights={}, aweights={})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"], kwargs["ddof"], kwargs["fweights"], kwargs["aweights"]
        )

        return code


class FunctionalCrossEntropyMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.cross_entropy({})
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class FunctionalSmoothL1LossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        API_TEMPLATE = "paddle.nn.functional.smooth_l1_loss({})"

        if "beta" in kwargs:
            kwargs["delta"] = kwargs.pop("beta")
            API_TEMPLATE = "paddle.nn.functional.smooth_l1_loss({})/" + str(
                kwargs["delta"]
            )

        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))

        return code


class FunctionalNllLossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.nll_loss({})
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class FunctionalMseLossMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "size_average" in kwargs:
            size_average = kwargs.pop("size_average")
            if "True" in size_average:
                size_average = True
            elif "False" in size_average:
                size_average = False
            else:
                size_average = None
        else:
            size_average = None

        if "reduce" in kwargs:
            reduce = kwargs.pop("reduce")
            if "True" in reduce:
                reduce = True
            elif "False" in reduce:
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

            kwargs["reduction"] = reduction

        if "target" in kwargs:
            kwargs["label"] = kwargs.pop("target")

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.nn.functional.mse_loss({})
            """
        )
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


class DiffMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "n" in kwargs and kwargs["n"] != "(1)":
            return None
        return GenericMatcher.generate_code(self, kwargs)


class ParameterMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_args_and_kwargs(args, kwargs)
        if "requires_grad" in kwargs:
            requires_grad_v = kwargs["requires_grad"]
        else:
            requires_grad_v = "True"

        API_TEMPLACE = textwrap.dedent(
            """
            {} = paddle.create_parameter(shape={}.shape, dtype=str({}.numpy().dtype), default_initializer=paddle.nn.initializer.Assign({}))
            {}.stop_gradient = not {}
            {}
            """
        )
        out = get_unique_name("out")
        code = API_TEMPLACE.format(
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
        if "eps" not in kwargs:
            epsilon = 1e-5
        else:
            epsilon = kwargs["eps"]

        if "track_running_stats" in kwargs:
            track_running_stats = kwargs["track_running_stats"]
        else:
            track_running_stats = True

        if "momentum" in kwargs:
            momentum = kwargs["momentum"]
        else:
            momentum = 0.1

        if "affine" in kwargs and "False" in kwargs["affine"]:
            API_TEMPLACE = textwrap.dedent(
                """
                {}(num_features={},
                    momentum=1-{},
                    epsilon={},
                    weight_attr=False,
                    bias_attr=False,
                    use_global_stats={})
                """
            )
        else:
            API_TEMPLACE = textwrap.dedent(
                """
                {}(num_features={},
                    momentum=1-{},
                    epsilon={},
                    weight_attr=None,
                    bias_attr=None,
                    use_global_stats={})
                """
            )
        code = API_TEMPLACE.format(
            self.get_paddle_api(),
            kwargs["num_features"],
            momentum,
            epsilon,
            track_running_stats,
        )
        return code


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
        self.parse_func(func)
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            import sys
            sys.path.append('{}')
            import paddle_aux
            {}.take({})
            """
        )
        code = API_TEMPLATE.format(
            self.get_aux_dir(),
            self.paddleClass,
            self.args_and_kwargs_to_str(new_args, new_kwargs),
        )
        return ast.parse(code.strip("\n")).body


class TensorSplitMatcher(BaseMatcher):
    def generate_aux_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def split(self, *args, **kwargs):
                if args:
                    if len(args)==1:
                        return paddle.split(self, self.shape[0]//args[0])
                    else:
                        return paddle.split(self, self.shape[args[1]]//args[0], args[1])
                elif kwargs:
                    if  "dim" in kwargs:
                        kwargs["axis"] = kwargs.pop("dim")
                        kwargs["num_or_sections"] = self.shape[kwargs["axis"]]//kwargs.pop("split_size")
                    else:
                        kwargs["num_or_sections"] = self.shape[0]//kwargs.pop("split_size")
                    return paddle.split(self, **kwargs)

            setattr(paddle.Tensor, 'split', split)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):

        self.write_aux_code()
        self.parse_func(func)
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            import sys
            sys.path.append('{}')
            import paddle_aux
            {}.split({})
            """
        )
        code = API_TEMPLATE.format(
            self.get_aux_dir(),
            self.paddleClass,
            self.args_and_kwargs_to_str(new_args, new_kwargs),
        )
        return ast.parse(code).body


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

        self.write_aux_code()
        self.parse_func(func)
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            import sys
            sys.path.append('{}')
            import paddle_aux
            {}.round({})
            """
        )
        code = API_TEMPLATE.format(
            self.get_aux_dir(),
            self.paddleClass,
            self.args_and_kwargs_to_str(new_args, new_kwargs),
        )
        return ast.parse(code).body
