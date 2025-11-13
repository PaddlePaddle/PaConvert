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
import astor
import textwrap

from paconvert.base import BaseMatcher
from paconvert.transformer.custom_op_transformer import CPP_EXTENSION_LIST  # noqa: F401
from paconvert.utils import get_unique_name

TypePromoteFunc = textwrap.dedent(
    """
    def TypePromote(x, y):
        TYPE_PROMOTE_DICT = {
            "INT16FP16": "float16",
            "INT16FP32": "float32",
            "INT16FP64": "float64",

            "INT32FP16": "float32",
            "INT32FP32": "float32",
            "INT32FP64": "float64",

            "INT64FP16": "float64",
            "INT64FP32": "float64",
            "INT64FP64": "float64",
        }
        if x.dtype.name + y.dtype.name in TYPE_PROMOTE_DICT:
            promote_type = TYPE_PROMOTE_DICT[x.dtype.name + y.dtype.name]
        elif y.dtype.name + x.dtype.name in TYPE_PROMOTE_DICT:
            promote_type = TYPE_PROMOTE_DICT[y.dtype.name + x.dtype.name]
        else:
            return x, y
        return x.astype(promote_type), y.astype(promote_type)
    """
)


def process_reduce_and_size_average(kwargs):
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


class GenericMatcher(BaseMatcher):
    def get_paddle_api(self):
        assert "paddle_api" in self.api_mapping_dict
        return super().get_paddle_api()

    def generate_code(self, kwargs):
        kwargs_change_value = self.api_mapping_dict.get("kwargs_change", {}).values()
        kwargs = self.change_kwargs(
            kwargs,
            [
                "layout",
                "device",
                "memory_format",
                "inplace",
                "generator",
                "non_blocking",
                "async",
            ],
        )
        kwargs = self.set_paddle_default_kwargs(kwargs)

        dtype_v = "None"
        if "dtype" in kwargs and "dtype" not in kwargs_change_value:
            dtype_v = kwargs.pop("dtype")

        pin_memory_v = "(False)"
        if "pin_memory" in kwargs and "pin_memory" not in kwargs_change_value:
            pin_memory_v = kwargs.pop("pin_memory")

        stop_gradient_v = "None"
        if "requires_grad" in kwargs and "requires_grad" not in kwargs_change_value:
            stop_gradient_v = "not " + kwargs.pop("requires_grad").strip("()")

        out_v = "None"
        if "out" in kwargs and "out" not in kwargs_change_value:
            out_v = kwargs.pop("out")

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

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


class SliceScatterMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" in kwargs:
            kwargs["x"] = kwargs.pop("input")
            x = kwargs["x"]
        else:
            x = self.paddleClass
        kwargs["value"] = kwargs.pop("src")

        API_TEMPLATE = textwrap.dedent(
            """
            shape = {}.shape
            axes, starts, ends, strides  = [0], [0], shape[0], [1]
            """
        )
        code = API_TEMPLATE.format(x)
        if "dim" in kwargs.keys():
            if "end" not in kwargs.keys():
                API_TEMPLATE = textwrap.dedent(
                    """
                    axes, ends = [{}], shape[{}]
                    """
                )
                code += API_TEMPLATE.format(kwargs["dim"], kwargs.pop("dim"))
            else:
                API_TEMPLATE = textwrap.dedent(
                    """
                    axes, ends = [{}], [{}]
                    """
                )
                code += API_TEMPLATE.format(kwargs.pop("dim"), kwargs.pop("end"))
        else:
            if "end" in kwargs.keys():
                API_TEMPLATE = textwrap.dedent(
                    """
                    ends = [{}]
                    """
                )
                code += API_TEMPLATE.format(kwargs.pop("end"))

        if "start" in kwargs.keys():
            API_TEMPLATE = textwrap.dedent(
                """
                starts = [{}]
                """
            )
            code += API_TEMPLATE.format(kwargs.pop("start"))
        if "step" in kwargs.keys():
            API_TEMPLATE = textwrap.dedent(
                """
                strides = [{}]
                """
            )
            code += API_TEMPLATE.format(kwargs.pop("step"))

        API_TEMPLATE = textwrap.dedent(
            """
            {}({}, axes=axes, starts=starts, ends=ends, strides=strides)
            """
        )
        code += API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class TensorFunc2PaddleFunc(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(
            kwargs,
            [
                "layout",
                "device",
                "memory_format",
                "inplace",
                "generator",
                "non_blocking",
                "async",
            ],
        )
        kwargs = self.set_paddle_default_kwargs(kwargs)
        code = "{}({}, {})".format(
            self.get_paddle_api(), self.paddleClass, self.kwargs_to_str(kwargs)
        )
        return code


class DetachMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.detach()".format(kwargs["input"])
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


class FSInitializeModelParallelMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "pipeline_length" not in kwargs:
            kwargs["pipeline_length"] = 1

        model_parallel_size = get_unique_name("model_parallel_size")
        data_parallel_size = get_unique_name("data_parallel_size")
        strategy = get_unique_name("strategy")

        API_TEMPLATE = textwrap.dedent(
            """
            {} = int(min(paddle.distributed.get_world_size(),{}))
            {} = int(paddle.distributed.get_world_size()/ ({} * {}))
            {} = paddle.distributed.fleet.DistributedStrategy()
            {}.hybrid_configs = dict(dp_degree={}, mp_degree={}, pp_degree={})
            paddle.distributed.fleet.init(is_collective=True, strategy={})
            """
        )
        code = API_TEMPLATE.format(
            model_parallel_size,
            kwargs["model_parallel_size_"],
            data_parallel_size,
            model_parallel_size,
            kwargs["pipeline_length"],
            strategy,
            strategy,
            data_parallel_size,
            model_parallel_size,
            kwargs["pipeline_length"],
            strategy,
        )
        return code


class FSModelParallelIsInitializedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = (
            "paddle.distributed.fleet.base.topology._HYBRID_PARALLEL_GROUP is not None"
        )
        return code


# TODO: why use Constant(0) ？When using constant initialization,
# regardless of whether it is initialized to 0, 0.01, or 1,
# there is no difference in the generated results. But when using
# random initialization, the generation effect of the model is
# semantically inferior. The reasons behind this phenomenon
# require further analysis.

# NOTE: The difference between ParallelEmbedding and VocaParallelEmbedding
# is the direction of segmentation. This mapping is not equivalent and
# requires additional consideration when converting its parameters.
class FSParallelEmbeddingMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "paddle.distributed.fleet.meta_parallel.\
                VocabParallelEmbedding(num_embeddings={},\
                embedding_dim={},weight_attr=paddle.nn.initializer.Constant(0))".format(
            kwargs["num_embeddings"], kwargs["embedding_dim"]
        )
        return code


class InferenceModeMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def empty_decorator(func):
                return func
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        if "mode" in kwargs:
            if kwargs["mode"] == "(False)":
                self.enable_utils_code()
                code = "empty_decorator"
            else:
                code = "paddle.no_grad()"
        else:
            code = "paddle.no_grad()"
        return code


# TODO: fix torch.atleast bug, which not support input list/tuple
class AtleastMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        if len(args) > 0 and isinstance(args[0], (ast.List, ast.Tuple)):
            code = "{}(*{})".format(self.get_paddle_api(), self.args_to_str(new_args))
        else:
            code = "{}({})".format(self.get_paddle_api(), self.args_to_str(new_args))
        return ast.parse(code).body


class EinopsTorchMatcher(BaseMatcher):
    def get_paddle_api(self):
        return self.torch_api.replace("einops.layers.torch", "einops.layers.paddle")

    def get_paddle_nodes(self, args, kwargs):
        return ChangeAPIMatcher.get_paddle_nodes(self, args, kwargs)


# These APIs only change torch.* to paddle.*, not change any other thing
class NoNeedConvertMatcher(BaseMatcher):
    def get_paddle_api(self):
        assert "paddle_api" not in self.api_mapping_dict
        if self.paddle_api:
            return self.paddle_api
        else:
            return self.torch_api.replace("torch.", "paddle.")

    def get_paddle_nodes(self, args, kwargs):
        args = self.parse_args(args)
        kwargs = self.parse_kwargs(kwargs, allow_none=True)

        # temporary delete these unsupport args, which paddle does not support now
        for k in ["layout", "generator", "memory_format", "sparse_grad"]:
            if k in kwargs:
                kwargs.pop(k)

        code = "{}({})".format(
            self.get_paddle_api(), self.args_and_kwargs_to_str(args, kwargs)
        )
        return ast.parse(code).body

    def get_paddle_class_attribute_nodes(self, node):
        return "unchange"


# These APIs only change API name, but not change API args/kwargs
class ChangeAPIMatcher(BaseMatcher):
    def get_paddle_api(self):
        assert "paddle_api" in self.api_mapping_dict
        assert "unsupport_args" not in self.api_mapping_dict
        assert "kwargs_change" not in self.api_mapping_dict
        assert "paddle_default_kwargs" not in self.api_mapping_dict
        return super().get_paddle_api()

    def get_paddle_nodes(self, args, kwargs):
        args = self.parse_args(args)
        kwargs = self.parse_kwargs(kwargs, allow_none=True)

        # temporary delete these unsupport args, which paddle does not support now
        for k in ["layout", "generator", "memory_format"]:
            if k in kwargs:
                kwargs.pop(k)

        code = "{}({})".format(
            self.get_paddle_api(), self.args_and_kwargs_to_str(args, kwargs)
        )
        return ast.parse(code).body


class CheckPointMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        args = self.parse_args(args)
        kwargs = self.parse_kwargs(kwargs, allow_none=True)

        for k in ["context_fn", "determinism_check", "debug"]:
            if k in kwargs:
                kwargs.pop(k)

        code = "{}({})".format(
            self.get_paddle_api(), self.args_and_kwargs_to_str(args, kwargs)
        )
        return ast.parse(code).body


class HubLoadMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        args = self.parse_args(args)
        kwargs = self.parse_kwargs(kwargs, allow_none=True)

        if "repo_or_dir" in kwargs:
            kwargs["repo_dir"] = kwargs.pop("repo_or_dir")
        for k in ["trust_repo", "verbose", "skip_validation"]:
            if k in kwargs:
                kwargs.pop(k)

        code = "{}({})".format(
            self.get_paddle_api(), self.args_and_kwargs_to_str(args, kwargs)
        )
        return ast.parse(code).body


class TransformersGenericMatcher(BaseMatcher):
    def get_paddle_api(self):
        return self.torch_api.replace("transformers.", "paddleformers.transformers.")

    def get_paddle_nodes(self, args, kwargs):
        return ChangeAPIMatcher.get_paddle_nodes(self, args, kwargs)


class PaddleFlagMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            class PaddleFlag:
                cudnn_enabled = True
                cudnn_benchmark = False
                matmul_allow_tf32 = False
                cudnn_allow_tf32 = True
                cudnn_deterministic = False
            """
        )
        return CODE_TEMPLATE


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
        kwargs = self.change_kwargs(kwargs, ["generator"])
        kwargs = self.set_paddle_default_kwargs(kwargs)

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


class InitEyeMatcher(InitMatcher):
    def generate_code(self, kwargs):
        init_tensor = kwargs["tensor"]
        init_value = "paddle.eye({}.shape[0], {}.shape[1])".format(
            init_tensor, init_tensor
        )
        kwargs["value"] = init_value
        return super().generate_code(kwargs)


class TensorStrideMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" not in kwargs or kwargs["dim"] == "None":
            API_TEMPLATE = textwrap.dedent(
                """
                {}.get_strides()
                """
            )
            code = API_TEMPLATE.format(self.paddleClass)
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {}.get_strides()[{}]
                """
            )
            code = API_TEMPLATE.format(self.paddleClass, kwargs["dim"])
        return code


class TensorToSparseCooMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            {}.to_sparse_coo(len({}.shape))
            """
        )
        code = API_TEMPLATE.format(self.paddleClass, self.paddleClass)
        return code


class TensorNbytesMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "{}.size * {}.element_size()".format(self.paddleClass, self.paddleClass)
        return ast.parse(code).body


class DimOrderMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            tuple([i for i in range(len({}.shape))])
            """
        )
        code = API_TEMPLATE.format(self.paddleClass)
        return code


class TRFMPreTrainedTokenizerMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import paddleformers
            original_encode = paddleformers.transformers.tokenizer_utils_base.PretrainedTokenizerBase.encode
            def _encode(self, *args, **kwargs):
                return original_encode(self, *args, **kwargs)["input_ids"]
            setattr(paddleformers.transformers.tokenizer_utils_base.PretrainedTokenizerBase, "encode", _encode)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return GenericMatcher.generate_code(self, kwargs)


class TRFMPreTrainedModelMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            from typing import Optional
            import paddleformers
            def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
                assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
                head_mask = head_mask.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
                return head_mask

            def _get_head_mask(
                self,
                head_mask: Optional[paddle.Tensor],
                num_hidden_layers: int,
                is_attention_chunked: bool = False,
            ):
                if head_mask is not None:
                    head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
                    if is_attention_chunked is True:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask
            setattr(paddleformers.transformers.model_utils.PretrainedModel, "get_head_mask", _get_head_mask)

            original_generate = paddleformers.generation.utils.GenerationMixin.generate
            def _generate(self, input_ids, *args, **kwargs):
                return paddle.concat((input_ids, original_generate(self,input_ids, *args, **kwargs)[0]), axis=-1)
            setattr(paddleformers.generation.utils.GenerationMixin, "generate", _generate)

            setattr(paddleformers.transformers.model_utils.PretrainedModel, "device", None)

            def _post_init(self):
                if hasattr(self, "init_weights"):
                    self.init_weights()
                elif hasattr(self, "_init_weights"):
                    self._init_weights()
            setattr(paddleformers.transformers.model_utils.PretrainedModel, "post_init", _post_init)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_nodes(self, args, kwargs):
        self.enable_utils_code()
        return ChangeAPIMatcher.get_paddle_nodes(self, args, kwargs)


class SignalWindowsWatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        if "sym" in kwargs:
            kwargs["fftbins"] = "not " + kwargs.pop("sym")
        if "exponential" in self.torch_api:
            if "tau" in kwargs:
                new_kwargs["window"] = "('exponential', None, {})".format(
                    kwargs.pop("tau")
                )
            else:
                new_kwargs["window"] = "('exponential', None, 1.0)"
        if "gaussian" in self.torch_api:
            if "std" in kwargs:
                new_kwargs["window"] = "('gaussian', {})".format(kwargs.pop("std"))
            else:
                new_kwargs["window"] = "('gaussian', 1.0)"
        if "general_hamming" in self.torch_api:
            if "alpha" in kwargs:
                new_kwargs["window"] = "('general_hamming', {})".format(
                    kwargs.pop("alpha")
                )
            else:
                new_kwargs["window"] = "('general_hamming', 0.54)"
        if "general_cosine" in self.torch_api:
            new_kwargs["window"] = "('general_cosine', {})".format(kwargs.pop("a"))
        new_kwargs.update(kwargs)
        return GenericMatcher.generate_code(self, new_kwargs)


class Num2TensorBinaryWithAlphaMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        if "y" in kwargs:
            if "alpha" in kwargs:
                kwargs["y"] = "paddle.to_tensor({}*{})".format(
                    kwargs.pop("alpha"), kwargs.pop("y")
                )
            else:
                kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("y"))

        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs.pop("out")
            code = "paddle.assign({}({}), output={})".format(
                self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class TensorAddMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_add(self, *args, **kwargs):
                if "other" in kwargs:
                    y = kwargs["other"]
                elif "y" in kwargs:
                    y = kwargs["y"]
                else:
                    y = args[0]
                if "alpha" in kwargs:
                    alpha = kwargs["alpha"]
                    if alpha != 1:
                        if not isinstance(y, paddle.Tensor):
                            y = paddle.to_tensor(alpha * y)
                        else:
                            y = alpha * y
                else:
                    if not isinstance(y, paddle.Tensor):
                        y = paddle.to_tensor(y)
                return paddle.add(self, y)

            setattr(paddle.Tensor, "add", _Tensor_add)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "unchange"


class TensorSubtractMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_sub(self, *args, **kwargs):
                if "other" in kwargs:
                    y = kwargs["other"]
                elif "y" in kwargs:
                    y = kwargs["y"]
                else:
                    y = args[0]
                if "alpha" in kwargs:
                    alpha = kwargs["alpha"]
                    if alpha != 1:
                        if not isinstance(y, paddle.Tensor):
                            y = paddle.to_tensor(alpha * y)
                        else:
                            y = alpha * y
                else:
                    if not isinstance(y, paddle.Tensor):
                        y = paddle.to_tensor(y)
                return paddle.subtract(self, y)

            setattr(paddle.Tensor, "sub", _Tensor_sub)
            setattr(paddle.Tensor, "subtract", _Tensor_sub)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "unchange"


class TensorMultiplyMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_mul(self, *args, **kwargs):
                if "other" in kwargs:
                    y = kwargs["other"]
                elif "y" in kwargs:
                    y = kwargs["y"]
                else:
                    y = args[0]

                if not isinstance(y, paddle.Tensor):
                    y = paddle.to_tensor(y)

                return paddle.multiply(self, y)

            setattr(paddle.Tensor, "mul", _Tensor_mul)
            setattr(paddle.Tensor, "multiply", _Tensor_mul)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "unchange"


class TensorDivideMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_div(self, *args, **kwargs):
                if "other" in kwargs:
                    y = kwargs["other"]
                elif "y" in kwargs:
                    y = kwargs["y"]
                else:
                    y = args[0]

                if not isinstance(y, paddle.Tensor):
                    y = paddle.to_tensor(y)

                res = paddle.divide(self, y)

                if "rounding_mode" in kwargs:
                    rounding_mode = kwargs["rounding_mode"]
                    if rounding_mode=="trunc":
                        res = paddle.trunc(res)
                    elif rounding_mode=="floor":
                        res = paddle.floor(res)

                return res

            setattr(paddle.Tensor, "div", _Tensor_div)
            setattr(paddle.Tensor, "divide", _Tensor_div)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "unchange"


class TransposeMatcher(BaseMatcher):
    def generate_utils_code(self):
        API_TEMPLATE = textwrap.dedent(
            """
            def dim2perm(ndim, dim0, dim1):
                perm = list(range(ndim))
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                return perm
            """
        )

        return API_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        dim0 = kwargs["dim0"] if "dim0" in kwargs else kwargs["axis0"]
        dim1 = kwargs["dim1"] if "dim1" in kwargs else kwargs["axis1"]
        API_TEMPLATE = textwrap.dedent(
            """
            {}(x={}, perm=dim2perm({}.ndim, {}, {}))
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(),
            kwargs["input"],
            kwargs["input"],
            dim0,
            dim1,
        )
        return code


class LoadMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["path"] = f'str({kwargs.pop("f")})'
        return GenericMatcher.generate_code(self, kwargs)


class TensorTransposeMatcher(BaseMatcher):
    def generate_utils_code(self):
        API_TEMPLATE = textwrap.dedent(
            """
            def dim2perm(ndim, dim0, dim1):
                perm = list(range(ndim))
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                return perm
            """
        )

        return API_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        dim0 = kwargs["dim0"] if "dim0" in kwargs else kwargs["axis0"]
        dim1 = kwargs["dim1"] if "dim1" in kwargs else kwargs["axis1"]
        API_TEMPLATE = textwrap.dedent(
            """
            {}(perm=dim2perm({}.ndim, {}, {}))
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(),
            self.paddleClass,
            dim0,
            dim1,
        )
        return code


class BroadcastShapesMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args) == 1 and isinstance(args[0], ast.Starred):
            return None
        new_args = self.parse_args(args)
        code = new_args[0]
        # Call the paddle.broadcast_shape multiple times
        for i in range(1, len(new_args)):
            code = "{}({}, {})".format(self.get_paddle_api(), code, new_args[i])
        return ast.parse(code).body


class TransformsPositiveDefiniteTransformMatcher(BaseMatcher):
    def generate_utils_code(self):
        API_TEMPLATE = textwrap.dedent(
            """
            class PositiveDefiniteTransform:
                def __call__(self, x):
                    x = x.tril(-1) + x.diagonal(axis1=-2, axis2=-1).exp().diag_embed()
                    return x @ x.T

                def inv(self, y):
                    y = paddle.linalg.cholesky(y)
                    return y.tril(-1) + y.diagonal(axis1=-2, axis2=-1).log().diag_embed()
            """
        )
        return API_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            PositiveDefiniteTransform()
            """
        )
        return API_TEMPLATE


class Is_InferenceMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        return "{}.stop_gradient".format(kwargs["input"])


class NumelMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        return "{}.size".format(kwargs["input"])


class AssertMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            assert {}, {}
            """
        )
        code = API_TEMPLATE.format(
            kwargs["condition"],
            kwargs["message"],
        )
        return code


class MakeTMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if "shape" not in kwargs:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).replace("\n", "")
            else:
                shape = self.parse_args(args)[0]
            kwargs = {"shape": str(shape).replace("'", ""), **kwargs}

        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        if "low" not in kwargs:
            kwargs["low"] = 0

        if "high" not in kwargs:
            kwargs["high"] = 1

        if "requires_grad" not in kwargs.keys():
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.uniform({}, dtype={}, min={}, max={}).to({})
                """
            )
            code = API_TEMPLATE.format(
                kwargs["shape"],
                kwargs["dtype"],
                kwargs["low"],
                kwargs["high"],
                kwargs["device"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                out = paddle.uniform({}, dtype={}, min={}, max={}).to({})
                out.stop_gradient = not {}
                out
                """
            )
            code = API_TEMPLATE.format(
                kwargs["shape"],
                kwargs["dtype"],
                kwargs["low"],
                kwargs["high"],
                kwargs["device"],
                kwargs["requires_grad"],
            )

        return ast.parse(code).body


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
                shape = astor.to_source(args[0].value).replace("\n", "")
            else:
                shape = self.parse_args(args)[0]

            kwargs = {"shape": str(shape).replace("'", ""), **kwargs}

        code = GenericMatcher.generate_code(self, kwargs)
        return ast.parse(code).body


class ModuleToMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "memory_format" in kwargs:
            kwargs.pop("memory_format")
        if "non_blocking" in kwargs:
            kwargs.pop("non_blocking")
        if "device" in kwargs:
            if "cuda" in kwargs["device"]:
                # case1: device = "cuda"
                # case2: device = "cuda:0"
                # case3: device = "cpu"
                # case4: device = "cpu:0"
                # case5: device = "cpu:0" if condition else "cuda:0"
                # case6: dev = xx, device = dev
                kwargs["device"] = kwargs["device"].replace("cuda", "gpu")

        code = "{}.to({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
        return code


class Device2StrBaseMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def device2str(type=None, index=None, *, device=None):
                type = device if device else type
                if isinstance(type, int):
                    type = f'gpu:{type}'
                elif isinstance(type, str):
                    if 'cuda' in type:
                        type = type.replace('cuda', 'gpu')
                    if 'cpu' in type:
                        type = 'cpu'
                    elif index is not None:
                        type = f'{type}:{index}'
                elif isinstance(type, paddle.CPUPlace) or (type is None):
                    type = 'cpu'
                elif isinstance(type, paddle.CUDAPlace):
                    type = f'gpu:{type.get_device_id()}'

                return type
            """
        )
        return CODE_TEMPLATE


class TorchDevice2StrMatcher(Device2StrBaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        # case1: torch.device(1)
        # case2: torch.device("cuda:1")
        # case3: torch.device("cuda", 1)

        # case4: torch.device(1 if cond else 0)
        # case5: torch.device("cuda:1" if cond else "cpu")
        # case6: torch.device("cuda" if cond else "cpu", 1 if cond else 0)

        # case7: a=1; torch.device(a)
        # case8: type="cuda:1"; torch.device(type)
        # case9: type="cuda"; index=1; torch.device(type, index)

        # case10: a=1 if cond else 0; torch.device(a)
        # case11: type="cuda:1" if cond else "cpu"; torch.device(type)
        # case12: type="cuda" if cond else "cpu"; index=1 if cond else 0; torch.device(type, index)
        self.enable_utils_code()
        args = self.parse_args(args)
        kwargs = self.parse_kwargs(kwargs)
        code = "device2str({})".format(self.args_and_kwargs_to_str(args, kwargs))
        return ast.parse(code).body


class GeluMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "approximate" in kwargs:
            approximate_v = kwargs.pop("approximate")
            if "none" in approximate_v:
                kwargs["approximate"] = "False"
            elif "tanh" in approximate_v:
                kwargs["approximate"] = "True"

        return GenericMatcher.generate_code(self, kwargs)


class ADVariableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            {}.stop_gradient = not {}
            {}
            """
        )

        if "requires_grad" in kwargs:
            return API_TEMPLATE.format(
                kwargs["data"], kwargs["requires_grad"], kwargs["data"]
            )
        return kwargs["data"]


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


class MPCpuCountMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            import os
            os.cpu_count()
            """
        )
        return API_TEMPLATE


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


class _MaxMinMatcherBase(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        kwargs_tmp = self.parse_kwargs(kwargs)
        if kwargs_tmp is None:
            return None

        call_maxmin = False
        call_maximinimum = False

        if "out" in kwargs_tmp:
            kwargs_tmp.pop("out")

        if len(kwargs_tmp) > 0:
            if "input" in kwargs_tmp and len(kwargs_tmp) == 1:
                call_maxmin = True
            if "dim" in kwargs_tmp or ("keepdim" in kwargs_tmp):
                call_maxmin = True
            if "other" in kwargs_tmp:
                call_maximinimum = True
        else:
            if len(args) != 2:
                call_maxmin = True
            elif isinstance(args[1], ast.Constant):
                call_maxmin = True

        paddle_api = self.get_paddle_api()

        # the case of two tensors
        if call_maximinimum:
            self.api_mapping_dict["args_list"] = ["input", "other", "*", "out"]
            new_kwargs = self.parse_args_and_kwargs(args, kwargs)
            new_kwargs["x"] = new_kwargs.pop("input")
            new_kwargs["y"] = new_kwargs.pop("other")

            if paddle_api == "paddle.min":
                paddle_api = "paddle.minimum"
            elif paddle_api == "paddle.max":
                paddle_api = "paddle.maximum"

            if "out" in new_kwargs:
                out_v = new_kwargs.pop("out")
                code = f"paddle.assign({paddle_api}({self.kwargs_to_str(new_kwargs)}), {out_v})"
            else:
                code = f"{paddle_api}({self.kwargs_to_str(new_kwargs)})"

            self.api_mapping_dict["args_list"] = ["input", "dim", "keepdim", "*", "out"]
            return ast.parse(code).body

        # the case of one tensor
        if call_maxmin:
            new_kwargs = self.parse_args_and_kwargs(args, kwargs)
            new_kwargs["x"] = new_kwargs.pop("input")
            if "dim" in new_kwargs:
                new_kwargs["axis"] = new_kwargs.pop("dim")

            paddle_arg_api = "paddle.argmin" if "min" in paddle_api else "paddle.argmax"

            if "axis" in new_kwargs:
                if "out" in new_kwargs:
                    out_v = new_kwargs.pop("out")
                    code = f"paddle.assign({paddle_api}({self.kwargs_to_str(new_kwargs)}), {out_v}[0]), paddle.assign({paddle_arg_api}({self.kwargs_to_str(new_kwargs)}), {out_v}[1])"
                else:
                    code = f"{paddle_api}({self.kwargs_to_str(new_kwargs)}), {paddle_arg_api}({self.kwargs_to_str(new_kwargs)})"
            else:
                code = f"{paddle_api}({self.kwargs_to_str(new_kwargs)})"

            return ast.parse(code).body

        self.enable_utils_code()
        if paddle_api == "paddle.min":
            self.set_paddle_api("paddle_min")
        elif paddle_api == "paddle.max":
            self.set_paddle_api("paddle_max")

        return ChangeAPIMatcher.get_paddle_nodes(self, args, kwargs)


class MaxMatcher(_MaxMinMatcherBase):
    def generate_utils_code(self):
        return textwrap.dedent(
            """
            def paddle_max(*args, **kwargs):
                if "input" in kwargs:
                    kwargs["x"] = kwargs.pop("input")

                out_v = None
                if "out" in kwargs:
                    out_v = kwargs.pop("out")

                if "other" in kwargs:
                    kwargs["y"] = kwargs.pop("other")
                    ret = paddle.maximum(*args, **kwargs)
                elif len(args)==2 and isinstance(args[1], paddle.Tensor):
                    ret = paddle.maximum(*args, **kwargs)
                else:
                    if "dim" in kwargs:
                        kwargs["axis"] = kwargs.pop("dim")

                    if "axis" in kwargs or len(args) >= 2:
                        if out_v:
                            ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                            paddle.assign(ret[0], out_v[0])
                            paddle.assign(ret[1], out_v[1])
                            return out_v
                        else:
                            ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                            return ret
                    else:
                        ret = paddle.max(*args, **kwargs)
                        return ret

                if out_v:
                    paddle.assign(ret, out_v)
                    return out_v
                else:
                    return ret
            """
        )


class MinMatcher(_MaxMinMatcherBase):
    def generate_utils_code(self):
        return textwrap.dedent(
            """
            def paddle_min(*args, **kwargs):
                if "input" in kwargs:
                    kwargs["x"] = kwargs.pop("input")

                out_v = None
                if "out" in kwargs:
                    out_v = kwargs.pop("out")

                if "other" in kwargs:
                    kwargs["y"] = kwargs.pop("other")
                    ret = paddle.minimum(*args, **kwargs)
                elif len(args)==2 and isinstance(args[1], paddle.Tensor):
                    ret = paddle.minimum(*args, **kwargs)
                else:
                    if "dim" in kwargs:
                        kwargs["axis"] = kwargs.pop("dim")

                    if "axis" in kwargs or len(args) >= 2:
                        if out_v:
                            ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                            paddle.assign(ret[0], out_v[0])
                            paddle.assign(ret[1], out_v[1])
                            return out_v
                        else:
                            ret = paddle.min(*args, **kwargs), paddle.argmin(*args, **kwargs)
                            return ret
                    else:
                        ret = paddle.min(*args, **kwargs)
                        return ret

                if out_v:
                    paddle.assign(ret, out_v)
                    return out_v
                else:
                    return ret
            """
        )


class TensorMaxMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_max(self, *args, **kwargs):
                if "other" in kwargs:
                    kwargs["y"] = kwargs.pop("other")
                    ret = paddle.maximum(self, *args, **kwargs)
                elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
                    ret = paddle.maximum(self, *args, **kwargs)
                else:
                    if "dim" in kwargs:
                        kwargs["axis"] = kwargs.pop("dim")

                    if "axis" in kwargs or len(args) >= 1:
                        ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
                    else:
                        ret = paddle.max(self, *args, **kwargs)

                return ret

            setattr(paddle.Tensor, "_max", _Tensor_max)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        if len(new_args) + len(new_kwargs) > 2:
            return "misidentify"

        if "other" in new_kwargs:
            new_kwargs["y"] = new_kwargs.pop("other")
            code = "{}.maximum({})".format(
                self.paddleClass, self.args_and_kwargs_to_str(new_args, new_kwargs)
            )
            return ast.parse(code).body

        if "dim" in new_kwargs:
            new_kwargs["axis"] = new_kwargs.pop("dim")
            kwargs_str = self.args_and_kwargs_to_str(new_args, new_kwargs)
            code = "({}.max({}), {}.argmax({}))".format(
                self.paddleClass, kwargs_str, self.paddleClass, kwargs_str
            )
            return ast.parse(code).body

        if len(new_args) > 1:
            kwargs_str = self.args_and_kwargs_to_str(new_args, new_kwargs)
            code = "({}.max({}), {}.argmax({}))".format(
                self.paddleClass, kwargs_str, self.paddleClass, kwargs_str
            )
            return ast.parse(code).body

        self.enable_utils_code()
        kwargs_str = self.args_and_kwargs_to_str(new_args, new_kwargs)
        code = "{}._max({})".format(self.paddleClass, kwargs_str)
        return ast.parse(code).body
        # return "unchange"


class TensorMinMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_min(self, *args, **kwargs):
                if "other" in kwargs:
                    kwargs["y"] = kwargs.pop("other")
                    ret = paddle.minimum(self, *args, **kwargs)
                elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
                    ret = paddle.minimum(self, *args, **kwargs)
                else:
                    if "dim" in kwargs:
                        kwargs["axis"] = kwargs.pop("dim")

                    if "axis" in kwargs or len(args) >= 1:
                        ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
                    else:
                        ret = paddle.min(self, *args, **kwargs)

                return ret

            setattr(paddle.Tensor, "_min", _Tensor_min)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        new_args = self.parse_args(args)
        new_kwargs = self.parse_kwargs(kwargs)
        if len(new_args) + len(new_kwargs) > 2:
            return "misidentify"

        if "other" in new_kwargs:
            new_kwargs["y"] = new_kwargs.pop("other")
            code = "{}.minimum({})".format(
                self.paddleClass, self.args_and_kwargs_to_str(new_args, new_kwargs)
            )
            return ast.parse(code).body

        if "dim" in new_kwargs:
            new_kwargs["axis"] = new_kwargs.pop("dim")
            kwargs_str = self.args_and_kwargs_to_str(new_args, new_kwargs)
            code = "({}.min({}), {}.argmin({}))".format(
                self.paddleClass, kwargs_str, self.paddleClass, kwargs_str
            )
            return ast.parse(code).body

        if len(new_args) > 1:
            kwargs_str = self.args_and_kwargs_to_str(new_args, new_kwargs)
            code = "({}.min({}), {}.argmin({}))".format(
                self.paddleClass, kwargs_str, self.paddleClass, kwargs_str
            )
            return ast.parse(code).body

        self.enable_utils_code()
        kwargs_str = self.args_and_kwargs_to_str(new_args, new_kwargs)
        code = "{}._min({})".format(self.paddleClass, kwargs_str)
        return ast.parse(code).body
        # return "unchange"


class EqualMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            {}({}).item()
            """
        )
        return API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class FAFlashAttnFuncMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        return GenericMatcher.generate_code(self, kwargs) + "[0]"


class FAFlashAttnUnpaddedFuncMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        if "scale" not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                import math
                assert paddle.device.cuda.get_device_capability()[0] >= 8, "Device capabilities should be at least 8"
                {}({})[0]
                """
            )
            kwargs["scale"] = "math.sqrt({}.shape[-1])".format(kwargs["query"])
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                assert paddle.device.cuda.get_device_capability()[0] >= 8, "Device capabilities should be at least 8"
                {}({})[0]
                """
            )
        return API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class GetLoggerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            import logging
            logging.getLogger({})
            """
        )
        return API_TEMPLATE.format(self.kwargs_to_str(kwargs))


class TRFMGenerationConfigMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        greedy_search_flag = False
        num_beans_value = 1
        if "do_sample" in kwargs:
            do_sample_value = kwargs["do_sample"]
            if do_sample_value != '"""False"""':
                greedy_search_flag = True
        if greedy_search_flag and "num_beans" in kwargs:
            num_beans_value = kwargs["num_beans"]
        if greedy_search_flag:
            greedy_search = f'"greedy_search" if {do_sample_value} else "sampling" if {num_beans_value} == 1 else "beam_search"'
            kwargs["greedy_search"] = greedy_search
        return f"{self.get_paddle_api()}({self.kwargs_to_str(kwargs)})"


class TensorMatcher(BaseMatcher):
    def get_paddle_api(self):
        if isinstance(
            self.transformer.parent_node, (ast.Tuple, ast.Index, ast.arg, ast.Subscript)
        ):
            return "paddle.Tensor"
        return super().get_paddle_api()

    def get_paddle_nodes(self, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None

        if "size" in kwargs:
            shape = kwargs.pop("size")
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif len(args) == 1 and isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).replace("\n", "")
            else:
                if len(args) == 0:
                    data = []
                else:
                    data = self.parse_args(args)[0]

                if (
                    "torch.IntTensor" == self.torch_api
                    or "torch.cuda.IntTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="int32")'.format(data)
                elif ("torch.ShortTensor" == self.torch_api) or (
                    "torch.cuda.ShortTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="int16")'.format(data)
                elif (
                    "torch.LongTensor" == self.torch_api
                    or "torch.cuda.LongTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="int64")'.format(data)
                elif ("torch.HalfTensor" == self.torch_api) or (
                    "torch.cuda.HalfTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="float16")'.format(data)
                elif (
                    "torch.FloatTensor" == self.torch_api
                    or "torch.cuda.FloatTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="float32")'.format(data)
                elif ("torch.DoubleTensor" == self.torch_api) or (
                    "torch.cuda.DoubleTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="float64")'.format(data)
                elif (
                    "torch.ByteTensor" == self.torch_api
                    or "torch.cuda.ByteTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="uint8")'.format(data)
                elif ("torch.BoolTensor" == self.torch_api) or (
                    "torch.cuda.BoolTensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="bool")'.format(data)
                elif ("torch.BFloat16Tensor" == self.torch_api) or (
                    "torch.cuda.BFloat16Tensor" == self.torch_api
                ):
                    code = 'paddle.tensor({}, dtype="bfloat16")'.format(data)
                else:
                    if len(args) > 0 and not isinstance(args[0], ast.Name):
                        code = 'paddle.tensor({}, dtype="float32")'.format(data)
                    else:
                        code = "paddle.tensor({})".format(data)
                return ast.parse(code).body

            shape = str(shape).replace("'", "")

        if (
            "torch.IntTensor" == self.torch_api
            or "torch.cuda.IntTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="int32")'.format(shape)
        elif (
            "torch.ShortTensor" == self.torch_api
            or "torch.cuda.ShortTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="int16")'.format(shape)
        elif (
            "torch.LongTensor" == self.torch_api
            or "torch.cuda.LongTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="int64")'.format(shape)
        elif (
            "torch.HalfTensor" == self.torch_api
            or "torch.cuda.HalfTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="float16")'.format(shape)
        elif (
            "torch.FloatTensor" == self.torch_api
            or "torch.cuda.FloatTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="float32")'.format(shape)
        elif (
            "torch.DoubleTensor" == self.torch_api
            or "torch.cuda.DoubleTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="float64")'.format(shape)
        elif (
            "torch.ByteTensor" == self.torch_api
            or "torch.cuda.ByteTensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="uint8")'.format(shape)
        elif ("torch.BFloat16Tensor" == self.torch_api) or (
            "torch.cuda.BFloat16Tensor" == self.torch_api
        ):
            code = 'paddle.empty(shape={}, dtype="bfloat16")'.format(shape)
        elif ("torch.BoolTensor" == self.torch_api) or (
            "torch.cuda.BoolTensor" == self.torch_api
        ):
            code = 'paddle.randint(0, 2, shape={}).astype("bool")'.format(shape)
        else:
            code = "paddle.empty(shape={})".format(shape)

        return ast.parse(code).body


class RandintMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        if "size" not in kwargs:
            new_kwargs["low"] = 0
            new_kwargs["high"] = kwargs.pop("low")
            new_kwargs["size"] = kwargs.pop("high")
        elif "high" not in kwargs:
            new_kwargs["low"] = 0
            new_kwargs["high"] = kwargs.pop("low")
        elif "low" not in kwargs:
            new_kwargs["low"] = 0

        new_kwargs.update(kwargs)
        return GenericMatcher.generate_code(self, new_kwargs)


class RandintLikeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        if "high" not in kwargs:
            new_kwargs["low"] = 0
            new_kwargs["high"] = kwargs.pop("low")
        elif "low" not in kwargs:
            new_kwargs["low"] = 0

        new_kwargs.update(kwargs)
        return GenericMatcher.generate_code(self, new_kwargs)


class ScatterReduceMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def get_reduce_type(type):
                map = {"sum": "add", "prod": "multiply"}
                if type == "sum" or type == "prod":
                    type = map[type]
                return type
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        kwargs["reduce"] = "get_reduce_type({})".format(kwargs["reduce"])
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


class TensorSizeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" in kwargs:
            code = "{}.shape[{}]".format(self.paddleClass, kwargs["dim"])
        else:
            code = "tuple({}.shape)".format(self.paddleClass)
        return code


class TensorRenameMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "columns" in kwargs:
            return "misidentify"

        return None


class TensorBF16Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="bfloat16")'.format(self.paddleClass)
        return code


class TensorBoolMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="bool")'.format(self.paddleClass)
        return code


class TensorByteMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="uint8")'.format(self.paddleClass)
        return code


class TensorCharMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="int8")'.format(self.paddleClass)
        return code


class TensorDoubleMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="float64")'.format(self.paddleClass)
        return code


class TensorFloatMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="float32")'.format(self.paddleClass)
        return code


class TensorFP16Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="float16")'.format(self.paddleClass)
        return code


class TensorIntMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="int32")'.format(self.paddleClass)
        return code


class TensorLongMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="int64")'.format(self.paddleClass)
        return code


class TensorShortMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="int16")'.format(self.paddleClass)
        return code


class TensorCfloatMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="complex64")'.format(self.paddleClass)
        return code


class TensorCdoubleMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = '{}.astype(dtype="complex128")'.format(self.paddleClass)
        return code


class TensorTypeAsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = f"{self.paddleClass}.astype(dtype={kwargs['other']}.dtype)"
        return code


class TensorTileMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None

        if "dims" in kwargs:
            kwargs = {"repeat_times": kwargs.pop("dims")}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                perm = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                perm = astor.to_source(args[0].value).replace("\n", "")
            else:
                perm = self.parse_args(args)[0]

            kwargs = {"repeat_times": str(perm).replace("'", "")}

        code = "{}.tile({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorNew_Matcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "size" in kwargs:
            kwargs["shape"] = kwargs.pop("size")
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).replace("\n", "")
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

        return ast.parse(code).body


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

        return code


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
        return code


class TorchTensorMatcher(BaseMatcher):
    def generate_code(self, kwargs):

        if "device" in kwargs:
            kwargs["place"] = kwargs.pop("device")
            if "cuda" in kwargs["place"]:
                kwargs["place"] = kwargs["place"].replace("cuda", "gpu")
        if "requires_grad" in kwargs:
            kwargs["stop_gradient"] = "not " + kwargs.pop("requires_grad").strip("()")

        if "pin_memory" in kwargs and kwargs.pop("pin_memory") == "(True)":
            kwargs["place"] = "paddle.CUDAPinnedPlace()"

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class CudaIsAvailableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}() >= 1".format(self.get_paddle_api())
        return code


class CudnnIsAvailableMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "bool(paddle.device.get_cudnn_version())"
        return code


class SplitMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def paddle_split(x, num_or_sections, axis=0):
                if isinstance(num_or_sections, int):
                    return paddle.split(x, x.shape[axis]//num_or_sections, axis)
                else:
                    return paddle.split(x, num_or_sections, axis)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        self.set_paddle_api("paddle_split")
        return GenericMatcher.generate_code(self, kwargs)


class RangeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "end" not in kwargs:
            kwargs["end"] = kwargs.pop("start")
        code = GenericMatcher.generate_code(self, kwargs)
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


class TensorCopy_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            {}({}, output={})
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(), kwargs["other"], self.paddleClass
        )
        return code


class TensorMaskedFillMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.where({}, {}, {})
            """
        )
        code = API_TEMPLATE.format(kwargs["mask"], self.paddleClass, kwargs["value"])
        return code


class TensorExpandMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None
        if "size" in kwargs:
            kwargs = {"shape": kwargs.pop("size")}
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                shape = self.parse_args(args)
            elif isinstance(args[0], ast.Starred):
                shape = astor.to_source(args[0].value).replace("\n", "")
            else:
                shape = self.parse_args(args)[0]

            kwargs = {"shape": str(shape).replace("'", "")}

        code = "{}.expand({})".format(self.paddleClass, self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class TensorRequiresGrad_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
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
        return code


class TensorTypeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if len(kwargs) == 0:
            code = f"str({self.paddleClass}.dtype)"
        else:
            # For torch.nn.Module.type, torch.nn.Module.type use torch.Tensor.type
            if "dst_type" in kwargs:
                code = f"{self.paddleClass}.astype({kwargs['dst_type']})"
            else:
                code = f"{self.paddleClass}.astype({kwargs['dtype']})"
        return code


class TensorIsCudaMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "{}.place.is_gpu_place()".format(self.paddleClass)
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

        return GenericMatcher.generate_code(self, kwargs)


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
            {}({}, [{}], [{}], [{}])
            """
        )
        start = get_unique_name("start")
        end = f"{start} + {kwargs['length']}"
        code = API_TEMPLATE.format(
            start,
            kwargs["input"],
            kwargs["dim"],
            kwargs["start"],
            kwargs["start"],
            kwargs["start"],
            self.get_paddle_api(),
            kwargs["input"],
            kwargs["dim"],
            start,
            end,
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


class AddCMul_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "value" not in kwargs:
            kwargs["value"] = 1

        API_TEMPLATE = textwrap.dedent(
            """
            {}.add_({} * {} * {})
            """
        )
        code = API_TEMPLATE.format(
            self.paddleClass, kwargs["value"], kwargs["tensor1"], kwargs["tensor2"]
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


class AddCDiv_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "value" not in kwargs:
            kwargs["value"] = 1

        API_TEMPLATE = textwrap.dedent(
            """
            {}.add_({} * {} / {})
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
            {}.astype("bool").item()
            """
        )
        code = API_TEMPLATE.format(kwargs["input"])

        return code


class TensorIndexCopyMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import numpy as np
            def _Tensor_index_copy_(self, dim, index, source):
                if dim == 0:
                    return self.scatter_(index, source)

                shape = self.shape

                new_index = []
                for i in range(0, np.prod(shape[:dim])):
                    new_index.append(index + i * len(index))
                new_index = paddle.concat(new_index)
                new_self = self.reshape_([-1] + shape[dim+1:])
                new_source = source.reshape([-1] + shape[dim+1:])

                return new_self.scatter_(new_index, new_source).reshape_(shape)

            setattr(paddle.Tensor, "index_copy_", _Tensor_index_copy_)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "unchange"


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


class ReverseMomentumMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "momentum" in kwargs:
            kwargs["momentum"] = f"1 - {kwargs.pop('momentum')}"
        return GenericMatcher.generate_code(self, kwargs)


class ReverseAsyncOpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "async_op" in kwargs:
            kwargs["sync_op"] = f"not {kwargs.pop('async_op')}"
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


class SizeMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        if len(args) == 0:
            code = "()"
        else:
            code = "tuple({})".format(astor.to_source(args[0]).replace("\n", ""))

        return ast.parse(code).body


class TensorToMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs):
        new_args = self.parse_args(args)
        for i, arg in enumerate(new_args):
            if "cuda" in arg:
                new_args[i] = arg.replace("cuda", "gpu")

        new_kwargs = self.parse_kwargs(kwargs)
        if new_kwargs is None:
            return None
        if "copy" in new_kwargs:
            new_kwargs.pop("copy")
        if "memory_format" in new_kwargs:
            new_kwargs.pop("memory_format")
        if "non_blocking" in new_kwargs:
            new_kwargs["blocking"] = "not " + new_kwargs.pop("non_blocking").strip("()")
        if "device" in new_kwargs:
            if "cuda" in new_kwargs["device"]:
                new_kwargs["device"] = new_kwargs["device"].replace("cuda", "gpu")

        code = "{}.to({})".format(
            self.paddleClass, self.args_and_kwargs_to_str(new_args, new_kwargs)
        )
        return ast.parse(code).body


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


class ErfC_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        API_TEMPLATE = textwrap.dedent(
            """
            paddle.erf_({}).multiply_(paddle.to_tensor(-1.)).add_(paddle.to_tensor(1.))
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


class XLogYMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "other" in kwargs:
            kwargs["other"] = "paddle.to_tensor({})".format(kwargs.pop("other"))

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


class XLogY_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "other" in kwargs:
            kwargs["other"] = f"paddle.to_tensor({kwargs.pop('other')})"

        API_TEMPLATE = textwrap.dedent(
            """
            {}.multiply_(paddle.log({}))
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


class Num2TensorBinaryConvertTypeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "other" in kwargs:
            kwargs[
                "y"
            ] = f"paddle.to_tensor({kwargs.pop('other')}, dtype={kwargs['input']}.dtype)"

        return GenericMatcher.generate_code(self, kwargs)


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


class AddMR_Matcher(BaseMatcher):
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

        if "beta" in kwargs:
            kwargs[
                "beta"
            ] = f"paddle.to_tensor({kwargs.pop('beta')}, dtype={kwargs['input']}.dtype)"
        else:
            kwargs["beta"] = f"paddle.to_tensor(1, dtype={kwargs['input']}.dtype)"

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1

        API_TEMPLATE = textwrap.dedent(
            """
            {}.multiply_({}).add_({}*{}({}, {}))
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"],
            kwargs["beta"],
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


class AddBmm_Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "beta" in kwargs:
            kwargs[
                "beta"
            ] = f"paddle.to_tensor({kwargs.pop('beta')}, dtype={kwargs['input']}.dtype)"
        else:
            kwargs["beta"] = f"paddle.to_tensor(1, dtype={kwargs['input']}.dtype)"

        if "alpha" not in kwargs:
            kwargs["alpha"] = 1

        API_TEMPLATE = textwrap.dedent(
            """
            {}.multiply_({}).add_({}*paddle.sum(paddle.bmm({}, {}), axis=0))
            """
        )
        code = API_TEMPLATE.format(
            kwargs["input"],
            kwargs["beta"],
            kwargs["alpha"],
            kwargs["batch1"],
            kwargs["batch2"],
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
        new_kwargs = {}
        for k in list(kwargs.keys()):
            if k == "input":
                new_kwargs["x"] = kwargs.pop(k)
            else:
                new_kwargs[k] = kwargs.pop(k)

        if "count_include_pad" in new_kwargs:
            new_kwargs["exclusive"] = "not " + new_kwargs.pop("count_include_pad")
        else:
            new_kwargs["exclusive"] = "False"

        API_TEMPLATE = textwrap.dedent(
            """
            {}({})
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(new_kwargs)
        )

        return code


class FSoftMinMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _get_softmin_dim(axis: int) -> int:
                if axis == 0 or axis == 1 or axis == 3:
                    return 0
                else:
                    return 1
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        kwargs["input"] = f"-{kwargs['input']}"
        if "dim" not in kwargs or kwargs["dim"] == "None":
            self.enable_utils_code()
            kwargs["dim"] = "_get_softmin_dim({}.ndim)".format(kwargs["input"])
            return GenericMatcher.generate_code(self, kwargs)
        else:
            return GenericMatcher.generate_code(self, kwargs)


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


class TriangularSolveMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        out_v = kwargs.pop("out") if "out" in kwargs else None
        new_kwargs = {}
        new_kwargs["x"] = kwargs.pop("A")
        new_kwargs["y"] = kwargs.pop("input")
        new_kwargs.update(kwargs)

        if out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}({}), {}[0]), paddle.assign({}, {}[1])
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(),
                self.kwargs_to_str(new_kwargs),
                out_v,
                new_kwargs["x"],
                out_v,
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {}({}), {}
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(), self.kwargs_to_str(new_kwargs), new_kwargs["x"]
            )

        return code


class TensorTriangularSolveMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["input"] = self.paddleClass
        return TriangularSolveMatcher.generate_code(self, kwargs)


class IndexAddMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "alpha" not in kwargs:
            kwargs["value"] = kwargs.pop("source")
        else:
            kwargs["value"] = "{} * {}".format(
                kwargs.pop("source"), kwargs.pop("alpha")
            )

        return GenericMatcher.generate_code(self, kwargs)


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
            kwargs["x"] = kwargs.pop("input")
        if "other" in kwargs:
            kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("other"))

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


class TensorDivideWithRoundingModeMatcher(BaseMatcher):
    def get_rounding_method(self, mode):
        if "trunc" in mode:
            return ".trunc_()"
        elif "floor" in mode:
            return ".floor_()"
        else:
            return ""

    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            {}({}){}
            """
        )

        if "other" in kwargs:
            kwargs["y"] = "paddle.to_tensor({})".format(kwargs.pop("other"))

        rounding_mode_v = "None"
        if "rounding_mode" in kwargs:
            rounding_mode_v = kwargs.pop("rounding_mode")
        rounding_method = self.get_rounding_method(rounding_mode_v)

        code = API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs), rounding_method
        )

        return code


class AllcloseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = GenericMatcher.generate_code(self, kwargs)
        code = "{}.item()".format(code)
        return code


class Assert_AllcloseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        msg = "''"
        if "msg" in kwargs:
            msg = kwargs.pop("msg")
        code = GenericMatcher.generate_code(self, kwargs)
        code = "assert {}.item(), {}".format(code, msg)
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


class TensorShapeMatcher(BaseMatcher):
    def get_paddle_class_attribute_nodes(self, node):
        self.parse_func(node)
        code = "tuple({}.shape)".format(self.paddleClass)
        return ast.parse(code).body


class TensorReshapeMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_reshape(self, *args, **kwargs):
                if args:
                    if len(args) == 1 and isinstance(args[0], (tuple, list)):
                        return paddle.reshape(self, args[0])
                    else:
                        return paddle.reshape(self, list(args))
                elif kwargs:
                    assert "shape" in kwargs
                    return paddle.reshape(self, shape=kwargs["shape"])

            setattr(paddle.Tensor, "reshape", _Tensor_reshape)
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
                self.enable_utils_code()
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

        return GenericMatcher.generate_code(self, kwargs)


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
                res[0], res[1].astype("float32")
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


class SpecialNdtrMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            (paddle.erf({}/paddle.sqrt(paddle.to_tensor(2.0)))-paddle.erf(paddle.to_tensor(-float("inf"))))/2
            """
        )
        code = API_TEMPLATE.format(kwargs["input"])
        if "out" in kwargs and kwargs["out"] != "None":
            code = "paddle.assign({}, output={})".format(code, kwargs["out"])

        return code


class LinalgInvExMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs["out"]
            API_TEMPLATE = textwrap.dedent(
                """
                out1 = paddle.linalg.inv({})
                out2 = paddle.zeros({}.shape[:-2], dtype="int32")
                paddle.assign(out1, output={}[0]), paddle.assign(out2, output={}[1])
                """
            )
            code = API_TEMPLATE.format(kwargs["A"], kwargs["A"], out_v, out_v)
            return code
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                (paddle.linalg.inv({}), paddle.zeros({}.shape[:-2], dtype="int32"))
                """
            )
            code = API_TEMPLATE.format(kwargs["A"], kwargs["A"])
            return code


class LinalgCholeskyExMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "upper" not in kwargs:
            kwargs["upper"] = False
        if "out" in kwargs and kwargs["out"] != "None":
            out_v = kwargs["out"]
            API_TEMPLATE = textwrap.dedent(
                """
                out1 = paddle.linalg.cholesky(x={}, upper={})
                out2 = paddle.zeros({}.shape[:-2], dtype="int32")
                paddle.assign(out1, output={}[0]), paddle.assign(out2, output={}[1])
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["upper"], kwargs["input"], out_v, out_v
            )
            return code
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                (paddle.linalg.cholesky(x={}, upper={}), paddle.zeros({}.shape[:-2], dtype="int32"))
                """
            )
            code = API_TEMPLATE.format(
                kwargs["input"], kwargs["upper"], kwargs["input"]
            )
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


class DoubleAssignMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        kwargs = self.set_paddle_default_kwargs(kwargs)
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
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return code


class TripleAssignMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        kwargs = self.set_paddle_default_kwargs(kwargs)
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
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class RoundMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass

        if "decimals" in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                {}((10**{}) * {}) / (10**{})
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(),
                kwargs["decimals"],
                kwargs["input"],
                kwargs["decimals"],
            )
        else:
            API_TEMPLATE = textwrap.dedent(
                """
                {}({})
                """
            )
            code = API_TEMPLATE.format(self.get_paddle_api(), kwargs["input"])

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

        if "bidirectional" in kwargs:
            if "(True)" == kwargs["bidirectional"]:
                kwargs["direction"] = "'bidirect'"
            kwargs.pop("bidirectional")

        return GenericMatcher.generate_code(self, kwargs)


class RNNBaseMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "batch_first" in kwargs:
            batch_first = kwargs.pop("batch_first")
        else:
            batch_first = False
        kwargs["time_major"] = f"not {batch_first}"

        direction = "'forward'"
        if "bidirectional" in kwargs:
            if "True" in kwargs["bidirectional"]:
                direction = "'bidirect'"
            kwargs.pop("bidirectional")
        kwargs["direction"] = direction

        return GenericMatcher.generate_code(self, kwargs)


class TensorTakeMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_take(self, *args, **kwargs):
                if args:
                    return paddle.take(self, *args)
                elif kwargs:
                    return paddle.take(self, **kwargs)

            setattr(paddle.Tensor, "take", _Tensor_take)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "unchange"


class TensorSplitMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_split(self, split_size, dim=0):
                if isinstance(split_size, int):
                    return paddle.split(self, self.shape[dim] // split_size, dim)
                else:
                    return paddle.split(self, split_size, dim)

            setattr(paddle.Tensor, "split", _Tensor_split)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        v = [v for v in kwargs.values()][0]
        if '"""' in v:
            return "misidentify"

        self.enable_utils_code()
        return "unchange"


class TensorRoundMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_round(self, decimals=None):
                if decimals:
                    x = paddle.abs(self)//(10**-decimals)*(10**-decimals)
                    return paddle.where(self<0, -x, x)
                return paddle.round(self)

            setattr(paddle.Tensor, "round", _Tensor_round)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        if len(kwargs) == 0:
            return "unchange"

        self.enable_utils_code()
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
            WARNING_TEMPLATE = textwrap.dedent(
                """
            paddle.utils.try_import("warnings").warn("Now, the return shape is inconsistent with torch when as_tuple is True")
            """
            )
            return WARNING_TEMPLATE + GenericMatcher.generate_code(self, kwargs)

        return GenericMatcher.generate_code(self, kwargs)


# This auxiliary function is not a completely equivalent transformation,
# which only implements the functional usage used by Qwen，not a complete
# implementation of flash_attn.layers.rotary.apply_rotary_emb_func.
class FAApplyRotaryEmbFuncMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def apply_rotary_position_embeddings(x, cos, sin):
                if not isinstance(cos, paddle.Tensor):
                    cos = paddle.to_tensor(cos)
                if not isinstance(sin, paddle.Tensor):
                    sin = paddle.to_tensor(sin)

                def _rotate_half(x):
                    from einops import rearrange

                    x = rearrange(x, "... (j d) -> ... j d", j=2)
                    x1, x2 = x.unbind(axis=-2)
                    return paddle.concat((-x2, x1), axis=-1)
                # [seq_len,rotary_dim/2] ==>[seq_len, rotary_dim]
                cos = paddle.concat([cos,cos],axis=-1)
                # [seq_len, rotary_dim] ==>[1,seq_len, 1,rotary_dim]
                cos=cos.unsqueeze(axis=1).unsqueeze(axis=0)
                # [seq_len,rotary_dim/2] ==>[seq_len, rotary_dim]
                sin = paddle.concat([sin,sin],axis=-1)
                # [seq_len, rotary_dim] ==>[1,seq_len, 1,rotary_dim]
                sin=sin.unsqueeze(axis=1).unsqueeze(axis=0)
                t_rot, t_pass = x[..., :cos.shape[-1]], x[..., cos.shape[-1]:]
                t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)

                return paddle.concat(x=(t_rot, t_pass), axis=-1)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            apply_rotary_position_embeddings({})
            """
        )
        return API_TEMPLATE.format(self.kwargs_to_str(kwargs))

    def get_paddle_api(self):
        self.enable_utils_code()
        return "apply_rotary_position_embeddings"


class FARmsNorm(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.incubate.nn.functional.fused_rms_norm({}, {}, paddle.zeros_like({}), {},len({}.shape)-1)[0]
            """
        )
        return API_TEMPLATE.format(
            kwargs["x"],
            kwargs["weight"],
            kwargs["weight"],
            kwargs["epsilon"],
            kwargs["x"],
        )


class NormMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "p" in kwargs and "nuc" in kwargs["p"]:
            return None
        return GenericMatcher.generate_code(self, kwargs)


class SortMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "input" not in kwargs:
            kwargs["x"] = self.paddleClass

        kwargs = self.change_kwargs(kwargs)
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
        return code


class TensorWhereMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            paddle.where({}, {}, {})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["condition"], self.paddleClass, kwargs["other"]
        )
        return code


class NTupleMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import collections
            from itertools import repeat
            def create_tuple_converter(n, name="parse"):
                def convert_to_tuple(x):
                    if isinstance(x, collections.abc.Iterable):
                        return tuple(x)
                    return tuple(repeat(x, n))

                convert_to_tuple.__name__ = name
                return convert_to_tuple
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        if "x" not in kwargs:
            API_TEMPLATE = textwrap.dedent(
                """
                create_tuple_converter({})
                """
            )
            code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        else:
            kwargs = self.set_paddle_default_kwargs(kwargs)
            API_TEMPLATE = textwrap.dedent(
                """
                create_tuple_converter({})({})
                """
            )
            code = API_TEMPLATE.format(kwargs["n"], kwargs["x"])

        return code


class Get_EnumMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def get_enum(reduction: str) -> int:
                if reduction == "none":
                    ret = 0
                elif reduction == "mean":
                    ret = 1
                elif reduction == "elementwise_mean":
                    warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
                    ret = 1
                elif reduction == "sum":
                    ret = 2
                else:
                    ret = -1
                    raise ValueError("{} is not a valid value for reduction".format(reduction))
                return ret
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            get_enum({})
            """
        )
        return API_TEMPLATE.format(kwargs["reduction"])


class SoftmaxMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _get_softmax_dim(axis: int) -> int:
                if axis == 0 or axis == 1 or axis == 3:
                    ret = 0
                else:
                    ret = 1
                return ret

            def _softmax_forward(self, x):
                if self._axis is None:
                    return paddle.nn.functional.softmax(x, _get_softmax_dim(x.ndim))
                return paddle.nn.functional.softmax(x, self._axis)
            setattr(paddle.nn.Softmax, "forward", _softmax_forward)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        if "dim" not in kwargs or kwargs["dim"] == "None":
            self.enable_utils_code()
        return GenericMatcher.generate_code(self, kwargs)


class LogSoftmaxMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _get_softmax_dim(axis: int) -> int:
                if axis == 0 or axis == 1 or axis == 3:
                    ret = 0
                else:
                    ret = 1
                return ret

            def _log_softmax_forward(self, x):
                if self._axis is None:
                    return paddle.nn.functional.log_softmax(x, _get_softmax_dim(x.ndim))
                return paddle.nn.functional.log_softmax(x, self._axis)
            setattr(paddle.nn.LogSoftmax, "forward", _log_softmax_forward)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        if "dim" not in kwargs or kwargs["dim"] == "None":
            self.enable_utils_code()
        return GenericMatcher.generate_code(self, kwargs)


class FSoftmaxMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _get_softmax_dim(axis: int) -> int:
                if axis == 0 or axis == 1 or axis == 3:
                    ret = 0
                else:
                    ret = 1
                return ret
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        if "dim" not in kwargs or kwargs["dim"] == "None":
            self.enable_utils_code()
            kwargs["dim"] = "_get_softmax_dim({}.ndim)".format(kwargs["input"])
        return GenericMatcher.generate_code(self, kwargs)


class SoftminMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        self.paddle_api = "Softmin"
        self.enable_utils_code()
        return GenericMatcher.generate_code(self, kwargs)

    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _get_softmax_dim(axis: int) -> int:
                if axis == 0 or axis == 1 or axis == 3:
                    ret = 0
                else:
                    ret = 1
                return ret

            class Softmin(paddle.nn.Softmax):
                def forward(self, x):
                    if self._axis is None:
                        return paddle.nn.functional.softmax(-x, _get_softmax_dim(x.ndim))
                    return paddle.nn.functional.softmax(-x, self._axis)
            """
        )
        return CODE_TEMPLATE


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


class LRSchedulerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        kwargs = self.set_paddle_default_kwargs(kwargs)

        optimizer = kwargs.pop("optimizer")
        API_TEMPLATE = textwrap.dedent(
            """
            tmp_lr = {}({})
            {}.set_lr_scheduler(tmp_lr)
            tmp_lr
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs), optimizer
        )

        return code


class Optim2LrSchedulerMatcher(LRSchedulerMatcher):
    def generate_code(self, kwargs):
        optimizer = kwargs["optimizer"]
        kwargs["learning_rate"] = "{}.get_lr()".format(optimizer)
        return super().generate_code(kwargs)


class ConstantLRMatcher(LRSchedulerMatcher):
    def generate_code(self, kwargs):
        optim = kwargs["optimizer"]
        total_iters = 5
        if "factor" in kwargs:
            factor = kwargs.pop("factor")
            kwargs["values"] = "[{}*{}.get_lr(), {}.get_lr()]".format(
                factor, optim, optim
            )
        else:
            kwargs["values"] = "[{}.get_lr()/3, {}.get_lr()]".format(optim, optim)

        if "total_iters" in kwargs:
            total_iters = kwargs.pop("total_iters")

        kwargs["boundaries"] = "[{}]".format(total_iters)
        return super().generate_code(kwargs)


class OneCycleLRMatcher(LRSchedulerMatcher):
    def generate_code(self, kwargs):
        if "total_steps" in kwargs and "None" not in kwargs["total_steps"]:
            pass
        else:
            steps_per_epoch = kwargs.pop("steps_per_epoch")
            epochs = kwargs.pop("epochs")
            kwargs["total_steps"] = "({})*({})".format(steps_per_epoch, epochs)

        if "final_div_factor" in kwargs:
            final_div_factor = kwargs.pop("final_div_factor")
            max_lr = kwargs["max_lr"]
            div_factor = kwargs["div_factor"]
            kwargs["end_learning_rate"] = "({}) * 1. /(({}) * {})".format(
                max_lr, div_factor, final_div_factor
            )

        return super().generate_code(kwargs)


class RequireDimMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" not in kwargs or "None" in kwargs["dim"]:
            return None

        return GenericMatcher.generate_code(self, kwargs)


class FunctionalLinearMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["weight"] = "{}.T".format(kwargs["weight"])
        return GenericMatcher.generate_code(self, kwargs)


class FunctionalBilinearMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "bias" in kwargs:
            kwargs["bias"] = "{}.unsqueeze(0)".format(kwargs["bias"])

        kwargs["x1"] = kwargs.pop("input1")
        kwargs["x2"] = kwargs.pop("input2")

        return "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class FunctionalOneHotMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = "{}.max().item() + 1".format(kwargs["input"])

        kwargs["x"] = kwargs.pop("input")

        return "{}({}).astype('int64')".format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs)
        )


class SizeAverageMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        process_reduce_and_size_average(kwargs)
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


class SymeigMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def convert_symeig(**kwargs):
                out_v = kwargs.pop("out", None)
                upper = kwargs.pop("upper", True)
                UPLO = "U" if upper else "L"
                eigenvectors = kwargs.pop("eigenvectors", False)
                if not eigenvectors:
                    result = (paddle.linalg.eigvalsh(kwargs["input"], UPLO=UPLO),
                              paddle.to_tensor([], dtype=paddle.complex64))
                else:
                    result = paddle.linalg.eigh(kwargs["input"], UPLO=UPLO)
                if out_v:
                    result = paddle.assign(result[0], out_v[0]), paddle.assign(result[1], out_v[1])
                return result
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        if "input" not in kwargs:
            kwargs["input"] = self.paddleClass
        return "convert_symeig({})".format(self.kwargs_to_str(kwargs))


class CanCastMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def can_cast(from_, to):
                can_cast_dict = {
                    "bfloat16": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": False,
                        "int8": False,
                        "int16": False,
                        "int32": False,
                        "int64": False,
                        "bool": False
                    },
                    "float16": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": False,
                        "int8": False,
                        "int16": False,
                        "int32": False,
                        "int64": False,
                        "bool": False,
                    },
                    "float32": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": False,
                        "int8": False,
                        "int16": False,
                        "int32": False,
                        "int64": False,
                        "bool": False,
                    },
                    "float64": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": False,
                        "int8": False,
                        "int16": False,
                        "int32": False,
                        "int64": False,
                        "bool": False,
                    },
                    "complex64": {
                        "bfloat16": False,
                        "float16": False,
                        "float32": False,
                        "float64": False,
                        "complex64": True,
                        "complex128": True,
                        "uint8": False,
                        "int8": False,
                        "int16": False,
                        "int32": False,
                        "int64": False,
                        "bool": False,
                    },
                    "complex128": {
                        "bfloat16": False,
                        "float16": False,
                        "float32": False,
                        "float64": False,
                        "complex64": True,
                        "complex128": True,
                        "uint8": False,
                        "int8": False,
                        "int16": False,
                        "int32": False,
                        "int64": False,
                        "bool": False,
                    },
                    "uint8": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": True,
                        "int8": True,
                        "int16": True,
                        "int32": True,
                        "int64": True,
                        "bool": False,
                    },
                    "int8": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": True,
                        "int8": True,
                        "int16": True,
                        "int32": True,
                        "int64": True,
                        "bool": False,
                    },
                    "int16": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": True,
                        "int8": True,
                        "int16": True,
                        "int32": True,
                        "int64": True,
                        "bool": False,
                    },
                    "int32": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": True,
                        "int8": True,
                        "int16": True,
                        "int32": True,
                        "int64": True,
                        "bool": False,
                    },
                    "int64": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": True,
                        "int8": True,
                        "int16": True,
                        "int32": True,
                        "int64": True,
                        "bool": False,
                    },
                    "bool": {
                        "bfloat16": True,
                        "float16": True,
                        "float32": True,
                        "float64": True,
                        "complex64": True,
                        "complex128": True,
                        "uint8": True,
                        "int8": True,
                        "int16": True,
                        "int32": True,
                        "int64": True,
                        "bool": True,
                    }
                }
                if isinstance(from_, paddle.base.core.DataType):
                    from_ = from_.name.lower()
                if isinstance(to, paddle.base.core.DataType):
                    to = to.name.lower()
                return can_cast_dict[from_][to]
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        return "can_cast(from_={}, to={})".format(kwargs["from_"], kwargs["to"])


class FloatPowerMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def cast_exponent(exponent):
                return exponent.cast(paddle.float64) if isinstance(exponent, paddle.Tensor) else exponent
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        if "input" in kwargs:
            code = "paddle.pow({}.cast(paddle.float64), cast_exponent({}))".format(
                kwargs["input"], kwargs["exponent"]
            )
            if "out" in kwargs:
                code = "paddle.assign({}, {})".format(code, kwargs["out"])
        else:
            code = "{}.cast(paddle.float64).pow(cast_exponent({}))".format(
                self.paddleClass, kwargs["exponent"]
            )
        return code


class FloatPowerInplaceMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        return "{}.cast_(paddle.float64).pow_({})".format(
            self.paddleClass, kwargs["exponent"]
        )


class ModuleGetSubMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = 'getattr({}, "{}")'.format(
            self.paddleClass,
            kwargs["target"].strip('"'),
        )
        return code


class TensorIsSignedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}.dtype not in [paddle.uint8]".format(self.paddleClass)
        return code


class TensorToBoolMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

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
        return ast.parse(code).body


class TensorInplaceReserveTypeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            _x_dtype_ = {}.dtype
            {}({}).cast_(_x_dtype_)
            """
        )

        convert_tensor = self.api_mapping_dict.get("convert_tensor", {})

        if "other" in kwargs:
            other_v = kwargs.pop("other")
            if "other" in convert_tensor:
                other_v = "paddle.to_tensor({})".format(other_v)
            kwargs["y"] = other_v

        code = API_TEMPLATE.format(
            self.paddleClass, self.get_paddle_api(), self.kwargs_to_str(kwargs)
        )

        return code


class Func2Attribute(BaseMatcher):
    def generate_code(self, kwargs):
        code = "{}".format(self.get_paddle_api())
        return code


class AllGatherObjectMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "group" not in kwargs:
            kwargs["group"] = None

        API_TEMPLATE = textwrap.dedent(
            """
                {}=[]
                {}(object_list={}, obj={}, group={})
            """
        )
        return API_TEMPLATE.format(
            kwargs["object_list"],
            self.get_paddle_api(),
            kwargs["object_list"],
            kwargs["obj"],
            kwargs["group"],
            self.kwargs_to_str(kwargs),
        )


class SetUpMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        is_torch_cpp_extension = False
        if "cmdclass" in kwargs:
            if "paddle.utils.cpp_extension.BuildExtension" in kwargs["cmdclass"]:
                is_torch_cpp_extension = True

        if not is_torch_cpp_extension:
            return "misidentify"

        kwargs.pop("cmdclass")
        global CPP_EXTENSION_LIST
        CPP_EXTENSION_LIST.append(kwargs["name"].strip('"'))
        return ast.parse(
            "paddle.utils.cpp_extension.setup({})".format(self.kwargs_to_str(kwargs))
        )


class SDPAttnMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "scale" in kwargs or "enable_gqa" in kwargs:
            return None
        query = kwargs.pop("query")
        key = kwargs.pop("key")
        value = kwargs.pop("value")
        API_TEMPLATE = textwrap.dedent(
            """
            {}({}.transpose([0, 2, 1, 3]), {}.transpose([0, 2, 1, 3]), {}.transpose([0, 2, 1, 3]), {}).transpose([0, 2, 1, 3])
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api(), query, key, value, self.kwargs_to_str(kwargs)
        )
        return code


class Is_PinnedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        code = f"'pinned' in str({self.paddleClass}.place)"

        return code


class Device2IntMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def device2int(device):
                if isinstance(device, str):
                    device = device.replace('cuda', 'gpu')
                    device = device.replace('gpu:', '')
                return int(device)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        if "device" in kwargs:
            kwargs["device_id"] = "device2int({})".format(kwargs.pop("device"))
        if "non_blocking" in kwargs:
            kwargs["blocking"] = f"not {kwargs.pop('non_blocking')}"
        return GenericMatcher.generate_code(self, kwargs)


class Device2StrMatcher(Device2StrBaseMatcher):
    def generate_code(self, kwargs):
        # paddle.set_device only recevice str type
        if "device" in kwargs:
            device = kwargs.pop("device")
            if device.strip("()").isdigit():
                # case 1: torch.set_default_device(0) => paddle.device.set_device("gpu:0")
                device = "'gpu:{}'".format(device.strip("()"))
            elif "cuda" in device:
                # case 2: torch.set_default_device("cuda:0") => paddle.device.set_device("gpu:0")
                # case 5: torch.set_default_device(device="cuda:0" if cond else "cpu:0")
                device = device.replace("cuda", "gpu")
            else:
                # case 3: torch.set_default_device("cpu") => paddle.device.set_device("cpu")
                # case 4: torch.set_default_device(device=0 if cond else 1)
                # case 6: num=2; torch.set_default_device(num)
                # case 7: device='cpu'; torch.set_default_device(device)
                # case 8: device='cuda:0'; torch.set_default_device(device)
                # case 9: device=torch.device('cpu'); torch.set_default_device(device)
                # case 10: device=torch.device('cuda:0'); torch.set_default_device(device)
                self.enable_utils_code()
                device = "device2str({})".format(device)
            kwargs["device"] = device
        return GenericMatcher.generate_code(self, kwargs)


class CudaStreamMatcher(Device2StrMatcher):
    def generate_code(self, kwargs):
        if "priority" in kwargs:
            kwargs["priority"] = "{}+2".format(kwargs.pop("priority"))
        return super().generate_code(kwargs)


class TensorViewMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def _Tensor_view(self, *args, **kwargs):
                if args:
                    if len(args)==1 and isinstance(args[0], (tuple, list, str, paddle.base.core.DataType)):
                        return paddle.view(self, args[0])
                    else:
                        return paddle.view(self, list(args))
                elif kwargs:
                    return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

            setattr(paddle.Tensor, 'view', _Tensor_view)
            """
        )
        return CODE_TEMPLATE

    def get_paddle_class_nodes(self, func, args, kwargs):
        if kwargs:
            if len(kwargs) == 1:
                self.enable_utils_code()
                return "unchange"

        if args:
            if len(args) == 1:
                if isinstance(args[0], (ast.Tuple, ast.List)):
                    return "unchange"
                if isinstance(args[0], (ast.Constant)) and isinstance(
                    args[0].value, str
                ):
                    return "unchange"

            self.enable_utils_code()
            return "unchange"

        return "misidentify"


class EmbeddingMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            class Embedding(paddle.nn.Embedding):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.padding_idx = self._padding_idx
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        self.set_paddle_api("Embedding")
        return GenericMatcher.generate_code(self, kwargs)


class OsEnvironGetMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "key" in kwargs:
            if kwargs["key"] == '"""WORLD_SIZE"""':
                code = "paddle.distributed.get_world_size()"
            elif kwargs["key"] == '"""LOCAL_RANK"""':
                code = "paddle.distributed.get_rank()"
            else:
                code = "misidentify"
        else:
            code = "misidentify"
        return code


class ZeroGradMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "set_to_none" in kwargs:
            set_to_none = kwargs.pop("set_to_none")
            kwargs["set_to_zero"] = f"(not {set_to_none})"

        return GenericMatcher.generate_code(self, kwargs)


# fix hard code
class SetDefaultTensorTypeMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if kwargs["t"] in ['"""torch.DoubleTensor"""', '"""torch.cuda.DoubleTensor"""']:
            kwargs["t"] = '"""float64"""'
        elif kwargs["t"] in ['"""torch.FloatTensor"""', '"""torch.cuda.FloatTensor"""']:
            kwargs["t"] = '"""float32"""'
        elif kwargs["t"] in ['"""torch.HalfTensor"""', '"""torch.cuda.HalfTensor"""']:
            kwargs["t"] = '"""float16"""'
        elif kwargs["t"] in [
            '"""torch.BFloat16Tensor"""',
            '"""torch.cuda.BFloat16Tensor"""',
        ]:
            kwargs["t"] = '"""bfloat16"""'
        return GenericMatcher.generate_code(self, kwargs)


class SimpleScalableVarMatcher(BaseMatcher):
    def get_scalable_var(self):
        args_list = self.api_mapping_dict.get("args_list", [])
        if len(args_list) != 1:
            return None
        arg_name = args_list[0]
        if not (arg_name.startswith("*") and len(arg_name) > 1):
            return None
        return arg_name[1:]

    def get_paddle_nodes(self, args, kwargs):
        var_arg_name = self.get_scalable_var()
        dest_var_arg_name = self.api_mapping_dict.get("kwargs_change", {}).get(
            var_arg_name, var_arg_name
        )
        if len(args) > 1:
            x = self.parse_args(args)
        else:
            if isinstance(args[0], ast.Starred):
                x = astor.to_source(args[0].value).replace("\n", "")
            else:
                x = self.parse_args(args)
        kwargs = {dest_var_arg_name: str(x).replace("'", "")}
        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class ScalableVarMatcher(BaseMatcher):
    def get_scalable_var(self):
        args_list = self.api_mapping_dict.get("args_list", [])
        if len(args_list) != 1:
            return None
        arg_name = args_list[0]
        if not (arg_name.startswith("*") and len(arg_name) > 1):
            return None
        return arg_name[1:]

    def get_paddle_nodes(self, args, kwargs):
        var_arg_name = self.get_scalable_var()
        if var_arg_name is None:
            return None

        dest_var_arg_name = self.api_mapping_dict.get("kwargs_change", {}).get(
            var_arg_name, var_arg_name
        )

        if var_arg_name in kwargs:
            kwargs[dest_var_arg_name] = kwargs.pop(var_arg_name)
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                dest_var_arg_value = self.parse_args(args)
            elif len(args) == 1 and isinstance(args[0], ast.Starred):
                dest_var_arg_value = astor.to_source(args[0].value).replace("\n", "")
            else:
                dest_var_arg_value = self.parse_args(args)[0]

            kwargs = {dest_var_arg_name: str(dest_var_arg_value).replace("'", "")}

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return ast.parse(code).body

    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        kwargs = self.parse_kwargs(kwargs)
        if kwargs is None:
            return None

        var_arg_name = self.get_scalable_var()
        if var_arg_name is None:
            return None

        dest_var_arg_name = self.api_mapping_dict.get("kwargs_change", {}).get(
            var_arg_name, var_arg_name
        )

        if var_arg_name in kwargs:
            kwargs[dest_var_arg_name] = kwargs.pop(var_arg_name)
        else:
            if len(args) > 1 or (len(args) == 1 and isinstance(args[0], ast.Constant)):
                dest_var_arg_value = self.parse_args(args)
            elif len(args) == 1 and isinstance(args[0], ast.Starred):
                dest_var_arg_value = astor.to_source(args[0].value).replace("\n", "")
            else:
                dest_var_arg_value = self.parse_args(args)[0]

            kwargs = {dest_var_arg_name: str(dest_var_arg_value).replace("'", "")}

        code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))
        return ast.parse(code).body


class Lu_unpackMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        kwargs = self.set_paddle_default_kwargs(kwargs)

        out_v = kwargs.pop("out", "None")
        if out_v != "None":
            out1 = "out1"
            out2 = "out2"
            out3 = "out3"
            if "unpack_ludata" in kwargs and kwargs["unpack_ludata"] == "(False)":
                out2 = "paddle.paddle.to_tensor([])"
                out3 = "paddle.paddle.to_tensor([])"
            if "unpack_pivots" in kwargs and kwargs["unpack_pivots"] == "(False)":
                out1 = "paddle.paddle.to_tensor([])"
            API_TEMPLATE = textwrap.dedent(
                """
                out1, out2, out3 = {}({})
                paddle.assign({}, {}[0]), paddle.assign({}, {}[1]), paddle.assign({}, {}[2])
                """
            )
            code = API_TEMPLATE.format(
                self.get_paddle_api(),
                self.kwargs_to_str(kwargs),
                out1,
                out_v,
                out2,
                out_v,
                out3,
                out_v,
            )
        else:
            code = "{}({})".format(self.get_paddle_api(), self.kwargs_to_str(kwargs))

        return code


class Linalg_qrMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs = self.change_kwargs(kwargs)
        kwargs = self.set_paddle_default_kwargs(kwargs)

        if "mode" in kwargs and kwargs["mode"] == '"""r"""':
            if "out" in kwargs:
                out_v = kwargs.pop("out")
                API_TEMPLATE = textwrap.dedent(
                    """
                    out1, out2 = (paddle.to_tensor([]),{}({}))
                    paddle.assign(out1, {}[0]), paddle.assign(out2, {}[1])
                    """
                )
                code = API_TEMPLATE.format(
                    self.get_paddle_api(), self.kwargs_to_str(kwargs), out_v, out_v
                )
            else:
                code = "(paddle.to_tensor([]),{}({}))".format(
                    self.get_paddle_api(), self.kwargs_to_str(kwargs)
                )
        else:
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
            else:
                code = "{}({})".format(
                    self.get_paddle_api(), self.kwargs_to_str(kwargs)
                )
        return code


class HistogramMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs_change = self.api_mapping_dict.get("kwargs_change", {})

        for k in kwargs_change:
            if k == "range" and k in kwargs:
                kwargs[kwargs_change[k][0]] = f"{kwargs['range']}[0]"
                kwargs[kwargs_change[k][1]] = f"{kwargs['range']}[1]"
                kwargs.pop(k)

        kwargs_bin_edges = kwargs.copy()
        if "weight" in kwargs_bin_edges:
            kwargs_bin_edges.pop("weight")
        if "density" in kwargs_bin_edges:
            kwargs_bin_edges.pop("density")

        if "out" in kwargs:
            out_v = kwargs.pop("out")
            kwargs_bin_edges.pop("out")
            if "Tensor" in self.torch_api:
                API_TEMPLATE = textwrap.dedent(
                    """
                    out1, out2 = {paddleclass}.histogram({kwargs}).cast({out_v}[0].dtype), {paddleclass}.histogram_bin_edges({kwargs_bin_edges}).cast({out_v}[1].dtype)
                    paddle.assign(out1, {out_v}[0]), paddle.assign(out2, {out_v}[1])
                    """
                )
                code = API_TEMPLATE.format(
                    paddleclass=self.paddleClass,
                    kwargs=self.kwargs_to_str(kwargs),
                    kwargs_bin_edges=self.kwargs_to_str(kwargs_bin_edges),
                    out_v=out_v,
                )
            else:
                API_TEMPLATE = textwrap.dedent(
                    """
                    out1, out2 = paddle.histogram({kwargs}).cast({out_v}[0].dtype), paddle.histogram_bin_edges({kwargs_bin_edges}).cast({out_v}[1].dtype)
                    paddle.assign(out1, {out_v}[0]), paddle.assign(out2, {out_v}[1])
                    """
                )
                code = API_TEMPLATE.format(
                    kwargs=self.kwargs_to_str(kwargs),
                    kwargs_bin_edges=self.kwargs_to_str(kwargs_bin_edges),
                    out_v=out_v,
                )
        else:
            if "Tensor" in self.torch_api:
                code = "{paddleclass}.histogram({kwargs}).cast({paddleclass}.dtype), {paddleclass}.histogram_bin_edges({kwargs_bin_edges}).cast({paddleclass}.dtype)".format(
                    paddleclass=self.paddleClass,
                    kwargs=self.kwargs_to_str(kwargs),
                    kwargs_bin_edges=self.kwargs_to_str(kwargs_bin_edges),
                )
            else:
                code = "paddle.histogram({kwargs}).cast({input}.dtype), paddle.histogram_bin_edges({kwargs_bin_edges}).cast({input}.dtype)".format(
                    kwargs=self.kwargs_to_str(kwargs),
                    kwargs_bin_edges=self.kwargs_to_str(kwargs_bin_edges),
                    input=kwargs["input"],
                )
        return code


class FromBufferMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            import numpy as np
            buffer = {0}
            dtype = {1}
            if isinstance(dtype, paddle.base.core.DataType):
                dtype = dtype.name.lower()
            paddle.to_tensor(np.frombuffer(np.array(buffer), dtype=dtype))
            """
        )
        code = API_TEMPLATE.format(kwargs["buffer"], kwargs["dtype"])

        return code


class RpcRemoteMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            class rpc_remote:
                def __init__(self, remote_obj):
                    self.remote = remote_obj

                def to_here(self):
                    return self.remote.wait()
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        kwargs["fn"] = kwargs.pop("func")
        kwargs = self.kwargs_to_str(kwargs)
        API_TEMPLATE = textwrap.dedent(
            """
            rpc_remote(paddle.distributed.rpc.rpc_async({}))
            """
        )
        return API_TEMPLATE.format(kwargs)


class GetNumThreadsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            import os
            os.getenv("CPU_NUM",1)
            """
        )
        code = API_TEMPLATE.format()
        return code


class GetNumInteropThreadsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        API_TEMPLATE = textwrap.dedent(
            """
            import os
            int(os.environ['OMP_NUM_THREADS'])
            """
        )
        code = API_TEMPLATE.format()
        return code


class SetNumInteropThreadsMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import os
            def _set_num_interop_threads(int):
                os.environ['OMP_NUM_THREADS'] = str(int)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            _set_num_interop_threads({})
            """
        )
        code = API_TEMPLATE.format(kwargs["int"])

        return code


class SetNumThreadsMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import os
            def _set_num_threads(int):
                os.environ['CPU_NUM'] = str(int)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            _set_num_threads({})
            """
        )
        code = API_TEMPLATE.format(kwargs["int"])

        return code


class CifarMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        if "root" in kwargs:
            root = kwargs.pop("root")
            data_file = (
                "cifar-100-python.tar.gz"
                if "Cifar100" in self.get_paddle_api()
                else "cifar-10-python.tar.gz"
            )
            kwargs["data_file"] = "os.path.join({}, '{}')".format(root, data_file)

        if "train" in kwargs:
            train_value = kwargs.pop("train").strip()
            if train_value == "(True)":
                kwargs["mode"] = "'train'"
            elif train_value == "(False)":
                kwargs["mode"] = "'test'"
            else:
                kwargs["mode"] = "'train' if {} else 'test'".format(train_value)

        API_TEMPLATE = textwrap.dedent(
            """
            import os
            {}({})
            """
        )
        return API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class MNISTMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        train_value = kwargs.pop("train", "(True)")
        if train_value == "(True)":
            kwargs["mode"] = "'train'"
        elif train_value == "(False)":
            kwargs["mode"] = "'test'"
        else:
            kwargs["mode"] = "'train' if {} else 'test'".format(train_value)

        if "root" in kwargs:
            root = kwargs.pop("root")
            dataset_name = (
                "FashionMNIST" if "FashionMNIST" in self.get_paddle_api() else "MNIST"
            )
            file_paths = {
                "train_image": f"{dataset_name}/raw/train-images-idx3-ubyte.gz",
                "train_label": f"{dataset_name}/raw/train-labels-idx1-ubyte.gz",
                "test_image": f"{dataset_name}/raw/t10k-images-idx3-ubyte.gz",
                "test_label": f"{dataset_name}/raw/t10k-labels-idx1-ubyte.gz",
            }
            if train_value == "(True)":
                kwargs[
                    "image_path"
                ] = f"os.path.join({root}, '{file_paths['train_image']}')"
                kwargs[
                    "label_path"
                ] = f"os.path.join({root}, '{file_paths['train_label']}')"
            elif train_value == "(False)":
                kwargs[
                    "image_path"
                ] = f"os.path.join({root}, '{file_paths['test_image']}')"
                kwargs[
                    "label_path"
                ] = f"os.path.join({root}, '{file_paths['test_label']}')"
            else:
                kwargs["image_path"] = (
                    f"os.path.join({root}, '{file_paths['train_image']}') if {train_value} else "
                    f"os.path.join({root}, '{file_paths['test_image']}')"
                )
                kwargs["label_path"] = (
                    f"os.path.join({root}, '{file_paths['train_label']}') if {train_value} else "
                    f"os.path.join({root}, '{file_paths['test_label']}')"
                )

        API_TEMPLATE = textwrap.dedent(
            """
            import os
            {}({})
            """
        )
        return API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class Flowers102Matcher(BaseMatcher):
    def generate_code(self, kwargs):
        split_value = kwargs.pop("split", '"""train"""')
        if split_value == '"""val"""':
            kwargs["mode"] = "'valid'"
        elif split_value == '"""test"""':
            kwargs["mode"] = "'test'"
        elif split_value == '"""train"""':
            kwargs["mode"] = "'train'"
        else:
            kwargs["mode"] = f"{split_value} if {split_value} != 'val' else 'valid'"

        if "root" in kwargs:
            root = kwargs.pop("root")
            file_paths = {
                "data_file": "flowers-102/102flowers.tgz",
                "label_file": "flowers-102/imagelabels.mat",
                "setid_file": "flowers-102/setid.mat",
            }
            kwargs["data_file"] = f"os.path.join({root}, '{file_paths['data_file']}')"
            kwargs["label_file"] = f"os.path.join({root}, '{file_paths['label_file']}')"
            kwargs["setid_file"] = f"os.path.join({root}, '{file_paths['setid_file']}')"

        API_TEMPLATE = textwrap.dedent(
            """
            import os
            {}({})
            """
        )
        return API_TEMPLATE.format(self.get_paddle_api(), self.kwargs_to_str(kwargs))


class VOCDetectionMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import os
            def VOCDetection(*args, **kwargs):
                root = kwargs.pop("root")
                year = kwargs.pop("year", "2012")
                if year != "2012":
                    raise ValueError("PaddlePaddle only supports VOC2012 dataset")
                image_set = kwargs.pop("image_set", "train")
                download = kwargs.pop("download", True)
                transform = kwargs.pop("transform", None)

                if image_set == "trainval":
                    mode = "train"
                elif image_set == "train":
                    mode = "test"
                elif image_set == "val":
                    mode = "valid"
                else:
                    raise ValueError("Only supports image_set in ['trainval', 'train', 'val']")

                data_file = os.path.join(root, "VOCtrainval_11-May-2012.tar")
                return paddle.vision.datasets.VOC2012(data_file=data_file, mode=mode, transform=transform, download=download, backend=None)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            VOCDetection({})
            """
        )
        code = API_TEMPLATE.format(self.kwargs_to_str(kwargs))
        return code


class DecodeJpegMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["x"] = kwargs.pop("input")
        device = kwargs.pop("device", "cpu")
        API_TEMPLATE = textwrap.dedent(
            """
            {}({}).to({})
            """
        )
        return API_TEMPLATE.format(
            self.get_paddle_api(), self.kwargs_to_str(kwargs), device
        )


class BoxesConvertMatcher(BaseMatcher):
    def generate_utils_code(self):
        api_name = self.get_paddle_api().split(".")[-1]
        CODE_TEMPLATE = textwrap.dedent(
            """
            def {}(*args, **kwargs):
                input = args[0] if len(args) > 0 else kwargs.get("input")
                boxes = args[1] if len(args) > 1 else kwargs.get("boxes")

                batch_size = input.shape[0]
                if isinstance(boxes, list):
                    boxes_num = [len(box) for box in boxes] + [0] * (batch_size - len(boxes))
                    boxes = paddle.concat(boxes) if boxes else paddle.zeros([0, 4])
                else:
                    boxes_num = [(boxes[:, 0] == i).sum() for i in range(batch_size)]
                    boxes = boxes[:, 1:]
                boxes_num = paddle.to_tensor(boxes_num, dtype="int32")

                kwargs["x"] = kwargs.pop("input")
                kwargs["boxes"] = boxes
                kwargs["boxes_num"] = boxes_num
                return paddle.vision.ops.{}(**kwargs)
            """
        ).format(api_name, api_name)
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            {}({})
            """
        )
        code = API_TEMPLATE.format(
            self.get_paddle_api().split(".")[-1], self.kwargs_to_str(kwargs)
        )
        return code


class WeightsMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        weights = kwargs.pop("weights", None)
        if weights and weights != "None":
            kwargs["pretrained"] = True
        return GenericMatcher.generate_code(self, kwargs)


class VGGMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        weights = kwargs.pop("weights", None)
        if weights and weights != "None":
            kwargs["pretrained"] = True
        kwargs["batch_norm"] = bool("bn" in self.torch_api)
        return GenericMatcher.generate_code(self, kwargs)


class GRUCellMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            class GRUCell(paddle.nn.GRUCell):
                def forward(self, inputs, states = None):
                    return super().forward(inputs, states)[0]
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        self.set_paddle_api("GRUCell")
        return GenericMatcher.generate_code(self, kwargs)


class LSTMCellMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            class LSTMCell(paddle.nn.LSTMCell):
                def forward(self, inputs, states = None):
                    return super().forward(inputs, states)[1]
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        self.set_paddle_api("LSTMCell")
        return GenericMatcher.generate_code(self, kwargs)


class RNNCellMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            class SimpleRNNCell(paddle.nn.SimpleRNNCell):
                def forward(self, inputs, states = None):
                    return super().forward(inputs, states)[0]
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        self.set_paddle_api("SimpleRNNCell")
        return GenericMatcher.generate_code(self, kwargs)


class TensorStftMatcher(BaseMatcher):
    def get_paddle_class_nodes(self, func, args, kwargs):
        self.parse_func(func)
        args = self.parse_args(args)
        kwargs = self.parse_kwargs(kwargs, allow_none=True)

        return_complex = kwargs.pop("return_complex", None)
        code = "{}({})".format(
            self.get_paddle_api(), self.args_and_kwargs_to_str(args, kwargs)
        )
        if return_complex and return_complex == "(False)":
            code += ".as_real()"
        return ast.parse(code).body

    def get_paddle_class_attribute_nodes(self, node):
        return "unchange"


class StftMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        return_complex = kwargs.pop("return_complex", None)
        code = GenericMatcher.generate_code(self, kwargs)
        if return_complex and return_complex == "(False)":
            return code + ".as_real()"
        else:
            return code


class Fractional_Max_pool2dMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        if "output_ratio" in kwargs:
            kwargs[
                "output_size"
            ] = "[int(paddle.shape({})[-2] * {}[0]),int(paddle.shape({})[-1] * {}[1])]".format(
                kwargs["input"],
                kwargs["output_ratio"],
                kwargs["input"],
                kwargs["output_ratio"],
            )
            kwargs.pop("output_ratio")
        for k in list(kwargs.keys()):
            if k == "input":
                new_kwargs["x"] = kwargs.pop(k)
            elif k == "return_indices":
                new_kwargs["return_mask"] = kwargs.pop(k)
            else:
                new_kwargs[k] = kwargs.pop(k)
        return GenericMatcher.generate_code(self, new_kwargs)


class Fractional_Max_pool3dMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        new_kwargs = {}
        if "output_ratio" in kwargs:
            kwargs[
                "output_size"
            ] = "[int(paddle.shape({})[-3] * {}[0]), int(paddle.shape({})[-2] * {}[1]),int(paddle.shape({})[-1] * {}[2])]".format(
                kwargs["input"],
                kwargs["output_ratio"],
                kwargs["input"],
                kwargs["output_ratio"],
                kwargs["input"],
                kwargs["output_ratio"],
            )
            kwargs.pop("output_ratio")
        for k in list(kwargs.keys()):
            if k == "input":
                new_kwargs["x"] = kwargs.pop(k)
            elif k == "return_indices":
                new_kwargs["return_mask"] = kwargs.pop(k)
            else:
                new_kwargs[k] = kwargs.pop(k)
        return GenericMatcher.generate_code(self, new_kwargs)


class OnnxExportMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def onnx_export(model,f):
                model = Logic()
                paddle.jit.to_static(model)
                last_dot_index = filename.rfind('.')
                if last_dot_index == -1:
                    path = f
                else:
                    path = f[:last_dot_index]
                return paddle.onnx.export(model, path)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            onnx_export({},{})
            """
        )
        code = API_TEMPLATE.format(kwargs["model"], kwargs["f"])

        return code


class SetPerProcessMemoryFractionMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import os
            def set_per_process_memory_fraction(fraction):
                os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = str(fraction)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            set_per_process_memory_fraction({})
            """
        )
        code = API_TEMPLATE.format(kwargs["fraction"])

        return code


class CudaGetRngStateMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            import os
            def cuda_get_rng_state(device):
                if isinstance(device, int):
                    return paddle.get_cuda_rng_state()[device]
                elif isinstance(device, str):
                    parts = device.split(":")
                    if len(parts) == 2 and parts[1].strip().isdigit():
                        return paddle.get_cuda_rng_state()[int(parts[1].strip())]
                    return paddle.get_cuda_rng_state()[0]
                elif isinstance(device, paddle.CUDAPlace):
                    return paddle.get_cuda_rng_state()[device.get_device_id()]

            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        if "device" not in kwargs:
            return "paddle.get_cuda_rng_state()[paddle.framework._current_expected_place().get_device_id()]"
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            cuda_get_rng_state({})
            """
        )
        code = API_TEMPLATE.format(kwargs["device"])

        return code


class ForeachMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def foreach_operator(func, tensors):
                return [func(x) for x in tensors]

            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            foreach_operator({}, {})
            """
        )
        code = API_TEMPLATE.format(self.get_paddle_api(), kwargs["self"])

        return code


class DistributedBackendMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        return "{}.lower()".format(kwargs["name"])


class JitSaveMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        kwargs["path"] = f"{kwargs.pop('f')}.rsplit('.', 1)[0]"
        return GenericMatcher.generate_code(self, kwargs)


class ForeachErfcMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def foreach_erfc(tensors):
                return [(1-paddle.erf(x)) for x in tensors]

            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            foreach_erfc({})
            """
        )
        code = API_TEMPLATE.format(kwargs["self"])
        return code


class ForeachErfc_Matcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def foreach_erfc_(tensors):
                return [paddle.assign(1-paddle.erf(x), x) for x in tensors]
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            foreach_erfc_({})
            """
        )
        code = API_TEMPLATE.format(kwargs["self"])
        return code


class ReduceScatterTensorMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def reduce_scatter_tensor(output, input, op, group, async_op):
                if async_op is not None:
                    async_op = not async_op
                world_size = paddle.distributed.get_world_size()
                input_list = []
                if input.shape[0] == world_size:
                    input_list = paddle.unstack(input, axis=0)
                else:
                    input_list = paddle.split(input, num_or_sections=world_size, axis=0)
                paddle.distributed.reduce_scatter(output, input_list, op, group, async_op)
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            reduce_scatter_tensor({},{},{},{},{})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["output"],
            kwargs["input"],
            kwargs.get("op"),
            kwargs.get("group"),
            kwargs.get("async_op"),
        )
        return code


class UtilsSetModuleMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def set_module(obj, mode):
                obj.__module__ = mode
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            set_module({},{})
            """
        )
        code = API_TEMPLATE.format(kwargs["obj"], kwargs["mod"])
        return code


class AllGatherIntoTensorMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def all_gather_into_tensor(output_tensor, input_tensor, group, async_op):
                if async_op is not None:
                    async_op = not async_op
                tensor_list = []
                paddle.distributed.all_gather(tensor_list=tensor_list, tensor=input_tensor, group=group, sync_op=async_op)
                if paddle.distributed.get_world_size() * input_tensor.shape[0] == output_tensor.shape[0]:
                    paddle.assign(paddle.concat(tensor_list, axis=0), output=output_tensor)
                else:
                    paddle.assign(paddle.stack(tensor_list, axis=0), output=output_tensor)
                return output_tensor
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            all_gather_into_tensor({},{},{},{})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["output_tensor"],
            kwargs["input_tensor"],
            kwargs.get("group"),
            kwargs.get("async_op"),
        )
        return code


class ForeachTensorMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def foreach_Tensor_(tensors,method_name):
                method_name = method_name.rpartition('.')[-1]
                return [getattr(x, method_name)() for x in tensors]
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            foreach_Tensor_({},"{}")
            """
        )
        code = API_TEMPLATE.format(kwargs["self"], self.get_paddle_api())
        return code


class LinalgLuSolveMatcher(BaseMatcher):
    def generate_utils_code(self):
        CODE_TEMPLATE = textwrap.dedent(
            """
            def linalg_lu_solve(LU, pivots, B, left=True, adjoint=False, out=None):
                if left:
                    trans = 'H' if adjoint else 'N'
                    result = paddle.linalg.lu_solve(lu=LU, pivots=pivots, b=B, trans=trans)
                else:
                    trans = 'N' if adjoint else 'H'
                    BH = B.conj().T
                    XH = paddle.linalg.lu_solve(lu=LU, pivots=pivots, b=BH, trans=trans)
                    result = XH.conj().T
                if out is not None:
                    return paddle.assign(result, output=out)
                else:
                    return result
            """
        )
        return CODE_TEMPLATE

    def generate_code(self, kwargs):
        self.enable_utils_code()
        API_TEMPLATE = textwrap.dedent(
            """
            linalg_lu_solve({},{},{},{},{},{})
            """
        )
        code = API_TEMPLATE.format(
            kwargs["LU"],
            kwargs["pivots"],
            kwargs["B"],
            kwargs.get("left", True),
            kwargs.get("adjoint", False),
            kwargs.get("out"),
        )
        return code
