# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#

import argparse
import json
import os
import re
import sys
import traceback

import pytest

sys.path.append(os.path.dirname(__file__) + "/../../tests")

import apibase

project_dir = os.path.join(os.path.dirname(__file__), "../..")
output_dir = os.path.dirname(__file__)

validate_whitelist = [
    r"^torch\.(cuda\.)?(\w*)Tensor",  # 特殊类构造函数，api_mapping.json 不合用，跳过
    r"^torch\.Size",  # api_mapping.json 不合用
    r"^torch\.nn\.ParameterList",  # 该文件没有测试 torch.nn.ParameterLisaa
    r"^torch\.utils\.cpp_extension\.BuildExtension",  # 该文件测试的 api 没有调用
    r"^torch\.utils\.data\.Sampler",  # 该文件测试时仅将 api 作为基类继承
    r"^torch\.utils\.data\.SequentialSampler",  # 该文件测试时仅将 api 作为基类继承
]

validate_whitelist_re = [re.compile(p) for p in validate_whitelist]

var_args_collector_aux_mapping = {
    "torch.Tensor.new_ones": "size",
    "torch.Tensor.permute": "dims",
    "torch.Tensor.repeat": "repeats",
    "torch.Tensor.reshape": "shape",
    "torch.Tensor.tile": "dims",
    "torch.Tensor.view": "size",
    "torch.empty": "size",
    "torch.ones": "size",
    "torch.rand": "size",
    "torch.randn": "size",
    "torch.zeros": "size",
    "torch.Tensor.new_empty": "size",
    "torch.Tensor.new_zeros": "size",
}

abstract_api_aux_set = {
    "torch.autograd.Function.forward",
    "torch.autograd.Function.backward",
    "torch.optim.Optimizer.step",
    "torch.utils.data.Dataset",
    "torch.nn.Module",
    "torch.autograd.Function",
    "torch.utils.data.IterableDataset",
}

overloadable_api_aux_set = {
    "torch.mean",
    "torch.Tensor.max",
    "torch.Tensor.to",
    "torch.trapz",
    "torch.prod",
    "torch.Tensor.var",
    "torch.Tensor.min",
    "torch.Tensor.sort",
    "torch.Tensor.std",
    "torch.trapezoid",
    "torch.normal",
    "torch.var_mean",
    "torch.std_mean",
    "torch.min",
    "torch.sort",
    "torch.max",
    "torch.searchsorted",
    "torch.cumulative_trapezoid",
    "torch.std",
    "torch.Tensor.scatter_",
    "torch.Tensor.scatter",
    "torch.scatter",
    "torch.Tensor.view",
    "torch.tensor_split",
    "torch.Tensor.dsplit",
    "torch.dsplit",
    "torch.Tensor.hsplit",
    "torch.hsplit",
}

cornercase_api_aux_dict = {
    "torch.Tensor.remainder": "keyword `divisor` or `other` is not supported unexpectedly",
    "torch.Tensor.type": "reserve dst_type arg for torch.nn.Module.type",
    "torch.Tensor.uniform_": 'keyword "from" is conflict with python keyword "from"',
    "torch.autograd.function.FunctionCtx.mark_non_differentiable": "expect only '*args' as arguments, so check is not supported",
    "torch.autograd.function.FunctionCtx.save_for_backward": "expect only '*tensors' as arguments, so check is not supported",
    "torch.chain_matmul": "this api will be deprecated and has var position args",
    "torch.clamp": "one of `min` and `max` must be specified.",
    "torch.from_numpy": "from_numpy() takes no keyword arguments",
    "torch.linalg.lstsq": "result shape dismatch",
    "torch.linalg.matrix_rank": "this api has deprecated arg `tol`",
    "torch.linalg.solve_triangular": 'keyword arg "upper" has no default value',
    "torch.nn.GRUCell": "paddle result has diff with pytorch result when all parameters use default value",
    "torch.nn.Identity": "this api accept any inputs but all is unused",
    "torch.nn.LSTMCell": "paddle result has diff with pytorch result when all parameters use default value",
    "torch.nn.functional.upsample": "only one of size or scale_factor should be defined",
    "torch.nn.functional.upsample_bilinear": "only one of size or scale_factor should be defined",
    "torch.nn.functional.upsample_nearest": "only one of size or scale_factor should be defined",
    "torch.profiler.schedule": "3 keyword args have no default value",
    "torch.randint_like": "this api has strange arg list, so `min_input_args` check is not supported",
    "torch.sparse_csr_tensor": "paddle must specified arg `shapes`",
    "torch.utils.cpp_extension.CUDAExtension": "args_list is configured by python built-in library",
    "torch.utils.cpp_extension.CppExtension": "args_list is configured by python built-in library",
    "torch.utils.dlpack.to_dlpack": 'arg "tensor" only accept position argument',
    "torch.var": "this api has breaking change in pytorch 2.0",
}


def get_test_cases(discovery_paths=["tests"]):
    # Collect the test cases
    monkeypatch = pytest.MonkeyPatch()
    unittest_data = {}
    pytest.main(
        discovery_paths + [],
        plugins=[RecordAPIRunPlugin(monkeypatch, unittest_data)],
    )
    return unittest_data


class RecordAPIRunPlugin:
    def __init__(self, monkeypatch, data):
        self.monkeypatch = monkeypatch
        self.collected_items = data
        originapi = apibase.APIBase.run

        def update_record(api, code, *, case_name=None, unsupport=False):
            sub_title = "code" if not unsupport else "unsupport"
            records = self.collected_items.get(api, {})
            sub_records = records.get(sub_title, {})
            sub_records.update({case_name: code})
            records.update({sub_title: sub_records})
            self.collected_items.update({api: records})

        def record_and_run(self, *args, **kwargs):
            caller_frame = traceback.extract_stack()[-2]
            caller_name = caller_frame.name

            pytorch_api = self.pytorch_api
            assert len(args) > 0
            pytorch_code = args[0]
            unsupport = kwargs.get("unsupport", False)
            expect_paddle_code = kwargs.get("expect_paddle_code", None)
            if expect_paddle_code is not None:
                unsupport = True

            update_record(
                pytorch_api, pytorch_code, case_name=caller_name, unsupport=unsupport
            )
            return originapi(self, *args, **kwargs)

        monkeypatch.setattr(apibase.APIBase, "run", record_and_run)

    def __del__(self):
        self.monkeypatch.undo()

    def pytest_collection_modifyitems(self, items):
        # Mount the monkeypatch
        collected = []
        for item in items:
            collected.append(item)

        items[:] = collected


def extract_api_name(api, code: str):
    api_seg = api.split(".")
    for i in range(len(api_seg)):
        api_nickname = ".".join(api_seg[i:])
        pattern = rf"\b{re.escape(api_nickname)}\b\("
        matched = re.search(pattern, code)
        if matched:
            return api_nickname
    else:
        raise ValueError(f"{api} not found in {repr(code)}")


def extract_params_from_invoking(api, code: str):
    args, kwargs = [], []
    api_name = extract_api_name(api, code)
    # 这个 pattern 假设 api 调用时一定跟着括号，不然会出错
    pattern = rf"\b{re.escape(api_name)}\b\("
    idx = re.search(pattern, code).start()
    assert idx >= 0, f"api_name {api_name} must exists."

    invoking_range = [idx + len(api_name), -1]

    code, idx = code[invoking_range[0] :], 0

    pair_stack = []
    key = None

    # 寻找参数列表的结束位置
    for i, c in enumerate(code):
        if c == "(" or c == "[" or c == "{":
            pair_stack.append(c)
            if len(pair_stack) == 1 and c == "(":
                idx = i + 1
        elif c == "=":
            # 如果是 "=="，这是合法的，不做等号处理
            if code[i + 1] == "=" or code[i - 1] == "=":
                pass
            # <=、>=、!= 也是合法的，不做等号处理
            elif code[i - 1] in "<>!":
                pass
            # 如果是单个等号，那就作为关键字参数处理
            elif len(pair_stack) == 1:
                key = code[idx:i]
                idx = i + 1
        elif c == "," or c == ")" or c == "]" or c == "}":
            if len(pair_stack) == 1:
                value = code[idx:i]
                if len(value.strip()) > 0:
                    if key is not None:
                        assert (
                            key not in kwargs
                        ), f"duplicated key {key} in {repr(code)}"
                        kwargs.append((key.strip(), value.strip()))
                        key = None
                    else:
                        args.append(value.strip())
                idx = i + 1
            if c == ")":
                assert (
                    len(pair_stack) > 0 and pair_stack[-1] == "("
                ), f"Unpaired in {repr(code)}"
                pair_stack.pop()
                if len(pair_stack) == 0:
                    invoking_range[1] = i + 1
                    break
            elif c == "]":
                assert pair_stack[-1] == "[", f"Unpaired in {repr(code)}"
                pair_stack.pop()
            elif c == "}":
                assert pair_stack[-1] == "{", f"Unpaired in {repr(code)}"
                pair_stack.pop()
        elif code[i : i + 7] == "lambda ":
            pair_stack.append("lambda")
        elif c == ":" and pair_stack[-1] == "lambda":
            pair_stack.pop()

    if len(pair_stack) != 0:
        raise ValueError(f"Unpaired in {repr(code)}")

    invoking_range[1] += invoking_range[0]

    return args, kwargs, invoking_range


def match_subsequence(pattern, obj):
    """
    验证 obj 是 pattern 的子序列
    """
    i, j = 0, 0

    while i < len(pattern) and j < len(obj):
        if pattern[i] == obj[j]:
            j += 1
        i += 1

    return j == len(obj)


def check_call_variety(test_data, api_mapping, *, api_alias={}, verbose=True):
    report = {}
    aux_detailed_data = {}

    for api, unittest_data in test_data.items():
        whitelist_skip = False
        for item in validate_whitelist_re:
            if item.match(api):
                whitelist_skip = True
                break
        if whitelist_skip:
            if verbose:
                print(f"SKIP: {api} is in whitelist, skip validations.")
            continue

        api_target = api_alias.get(api, api)
        if api_target not in api_mapping:
            if "code" not in unittest_data:
                if verbose:
                    print(f"SKIP: {api} is not prepared, skip validations.")
            else:
                if verbose:
                    print(f"WARNING: {api} not found in api_mapping")
            continue

        aux_detailed_case_data = {}
        aux_detailed_data_api = {
            "complex": False,
        }

        mapping_data = api_mapping[api_target]

        if "code" not in unittest_data:
            "如果 = 0，说明该 api 只是占位符"
            if len(mapping_data) > 0 and "Matcher" in mapping_data:
                if verbose:
                    print(f"WARNING: {api} has no unittest.")
            continue

        is_partial_support = (
            "unsupport" in unittest_data and len(unittest_data["unsupport"]) > 0
        ) or "unsupport_args" in mapping_data

        abstract = api_target in abstract_api_aux_set
        if abstract:
            if verbose:
                print(f"SKIP: {api} is abstract.")
            continue

        cornercase_exists = cornercase_api_aux_dict.get(api_target, None)
        if cornercase_exists:
            if verbose:
                print(f"SKIP: {api} has some corner cases: {cornercase_exists}.")

        if "Matcher" not in mapping_data:
            if verbose:
                print(f"WARNING: {api} has no mapping data 'Matcher'.")
            continue

        is_overloadable = api_target in overloadable_api_aux_set
        if is_overloadable:
            aux_detailed_data_api["complex"] = True

        position_args_checkable = True

        min_input_args = mapping_data.get("min_input_args", -1)
        aux_detailed_data_api["min_input_args"] = min_input_args

        args_list_full = mapping_data.get("args_list", [])

        var_arg_name = None
        var_kwarg_name = None

        var_args_collector = var_args_collector_aux_mapping.get(api_target, None)

        _args_list_position_end = len(args_list_full)
        args_list_full_keyword = []
        args_list_full_positional = []
        is_token = lambda x: x.isalpha() or x == "_"
        for i, arg in enumerate(args_list_full):
            if arg.startswith("*"):
                # 首个星号之前的是位置参数，之后的是关键字参数
                _args_list_position_end = min(_args_list_position_end, i)
                if arg.startswith("**"):
                    if len(arg) > 2 and not is_token(arg[2]):
                        raise ValueError(f'{api} has unexpected arg "{arg}".')
                    # 允许匿名可变参数列表，如 **kwargs 或 **
                    var_kwarg_name = arg[2:]
                else:
                    if var_arg_name is not None:
                        if len(arg[1:]) > 0:
                            raise ValueError(
                                f'{api} has duplicated var_args_collector "{var_arg_name}" and "{arg[1:]}"'
                            )
                    else:
                        var_arg_name = arg[1:]

                    if var_arg_name == var_args_collector:
                        if var_arg_name not in args_list_full_keyword:
                            args_list_full_keyword.append(var_arg_name)
                        args_list_full_positional.append(var_arg_name)
            elif is_token(arg[0]):
                args_list_full_keyword.append(arg)
                if i < _args_list_position_end:
                    args_list_full_positional.append(arg)

                if arg == var_args_collector:
                    args_list_full_positional.append(arg)
                    var_arg_name = arg
            elif arg == "/":
                position_args_checkable = True
                aux_detailed_data_api["complex"] = True
            else:
                raise ValueError(f'{api} has unexpected arg "{arg}".')

        if var_arg_name is not None and var_args_collector is not None:
            aux_detailed_data_api["complex"] = True
        if var_kwarg_name is not None:
            aux_detailed_data_api["complex"] = True

        # 这里只移除了不支持的参数，但是事实上，移除不支持的参数会影响其他检查，如
        # (a, b, c, d) 中移除了 (c)，那么不指定关键字最多只能传入 a、b
        unsupport_args = mapping_data.get("unsupport_args", [])
        if len(unsupport_args) > 0:
            aux_detailed_data_api["complex"] = True
        args_list = [arg for arg in args_list_full if arg not in unsupport_args]

        __pargs_end = len(args_list_full_positional)
        for i, arg in enumerate(args_list_full_positional):
            if arg in unsupport_args:
                __pargs_end = min(__pargs_end, i)
            elif i > __pargs_end:
                position_args_checkable = False

        args_list_positional = args_list_full_positional[:__pargs_end]
        aux_detailed_data_api["position_args_list"] = args_list_positional

        args_list_keyword = [
            arg for arg in args_list_full_keyword if arg not in unsupport_args
        ]
        aux_detailed_data_api["keyword_args_list"] = args_list_keyword

        all_args = False
        all_kwargs = False
        not_subsequence = False
        all_default = False if "min_input_args" in mapping_data else None

        for case_name, code in unittest_data["code"].items():
            try:
                api_name = extract_api_name(api, code)
                args, kwargs, _ = extract_params_from_invoking(api, code)
            except Exception:
                print(f'Error when extract params from invoking "{api}"')
                print(traceback.format_exc())
                exit(-1)

            aux_detailed_case_data[case_name] = {}

            # 检查 kwargs 是否符合预设的 args_list
            # 条件是，如果没有 **kwargs 参数，那么 kwargs 的 key 必须是 args_list 的子集

            if var_kwarg_name is None and len(kwargs) > 0:
                for k, v in kwargs:
                    if k not in args_list_full_keyword:
                        if not is_overloadable:
                            raise ValueError(
                                f"{api}(*{args}, **{kwargs}) has unexpected keyword argument '{k}'."
                            )
                        else:
                            if verbose:
                                print(
                                    f"WARNING: {api} has overload keyword argument '{k}'."
                                )
                            break

            # 如果没有 *arg，args 的长度必须小于等于 args_list_positional 的长度
            support_var_args = var_arg_name is not None and len(var_arg_name) > 0
            if not support_var_args and len(args) > len(args_list_positional):
                raise ValueError(
                    f"{api}(*{args}, **{kwargs}) has too many position arguments, args_list={args_list_keyword}"
                )

            if len(args) == min_input_args:
                if len(kwargs) == 0:
                    all_default = True
                    aux_detailed_case_data[case_name]["all_default"] = True
            elif len(args) < min_input_args:
                if len(kwargs) + len(args) == min_input_args and len(args) == len(
                    args_list_positional
                ):
                    all_default = True
                    aux_detailed_case_data[case_name]["all_default"] = True
                elif len(kwargs) + len(args) < min_input_args:
                    raise ValueError(
                        f"{api}(*{args}, **{kwargs}) has too few arguments, args_list={args_list_keyword}"
                    )

            if len(args) == len(args_list_positional):
                all_args = True
            elif len(args) >= len(args_list_positional) and support_var_args:
                all_args = True

            if len(args) + len(kwargs) == len(args_list_keyword):
                aux_detailed_case_data[case_name]["all_*args"] = True

            keys = [k[0].strip() for k in kwargs]
            if len(keys) == len(args_list_keyword) and match_subsequence(
                args_list_keyword, keys
            ):
                all_kwargs = True
                aux_detailed_case_data[case_name]["all_kwargs"] = True

            if len(args_list_keyword) <= 1 or not match_subsequence(
                args_list_keyword, keys
            ):
                not_subsequence = True

        # if api == "torch.optim.Adagrad":
        #     print(position_args_checkable, keys)
        #     print(all_args, all_kwargs, not_subsequence, code)

        if not position_args_checkable:
            all_args = None

        aux_detailed_data_api["cases"] = aux_detailed_case_data
        aux_detailed_data_api.update(
            {
                "all args": all_args,
                "all kwargs": all_kwargs,
                "kwargs out of order": not_subsequence,
                "all default": all_default,
            }
        )
        aux_detailed_data[api] = aux_detailed_data_api

        if all_args and all_kwargs and not_subsequence and all_default is True:
            continue

        report[api] = {
            "all args": all_args,
            "all kwargs": all_kwargs,
            "kwargs out of order": not_subsequence,
            "all default": all_default,
            "partial support": is_partial_support,
            "overloadable": is_overloadable,
            "corner case": cornercase_exists,
        }

        if verbose:
            if position_args_checkable:
                if not all_args:
                    print(
                        f"INFO: {api} has no unittest with all arguments without keyword."
                    )
            else:
                print(f"INFO: {api} has some position args is not supported.")
            if not all_kwargs:
                print(f"INFO: {api} has no unittest with all arguments with keyword.")
            if not not_subsequence and len(args_list) > 1:
                print(
                    f"INFO: {api} has no unittest with keyword arguments out of order."
                )
            if not all_default:
                if all_default is False:
                    print(
                        f"INFO: {api} has no unittest with all default arguments, min_input_args={min_input_args}."
                    )
                else:
                    print(f"INFO: {api} do not have configured 'min_input_args'.")

    return report, aux_detailed_data


def simple_map_api_url(api, *, api_alias={}):
    api = api_alias.get(api, api)
    if api.startswith("torch.distributions."):
        if api.endswith("Transform"):
            api = api.replace(
                "torch.distributions.", "torch.distributions.transforms.", 1
            )
        else:
            api = api.replace("torch.distributions.", "", 1).lower()

        return f"https://pytorch.org/docs/stable/distributions.html#{api}"
    return f"https://pytorch.org/docs/stable/generated/{api}.html"


TESTCASE_NAME_PATTERN = re.compile(
    r"^def (?P<case_name>(?P<hidden>_?)test_case_(?P<case_id>\d+))\(\):.*$"
)
TESTCASE_END_PATTERN = re.compile(r"^[ \t]+obj\.run\(.*")


# 尝试自动修复单个 api
def autofix_single_api(file_path, aux_detailed_data):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return

    assert len(aux_detailed_data) == 1
    api = list(aux_detailed_data.keys())[0]
    data = aux_detailed_data[api]

    if (
        data.get("all args", False) is True
        and data.get("all kwargs", False) is True
        and data.get("kwargs out of order", False) is True
        and data.get("all default", False) is True
    ):
        return

    if data.get("complex", False) is True:
        print(f'api {api} in "{file_path}" is too complex to auto fix.')
        return

    with open(file_path, "r") as f:
        lines = f.readlines()

    case_lines = {}
    state = 0
    for i, l in enumerate(lines):
        if state == 0:
            matched = TESTCASE_NAME_PATTERN.match(l)
            if matched:
                case_id = int(matched["case_id"])
                case_name = matched["case_name"]
                case_lines[case_name] = [i, len(lines)]
                state = 1
        elif state == 1:
            if TESTCASE_END_PATTERN.match(l):
                state = 2
        elif state == 2:
            if len(l.rstrip(" \n")) > 0:
                pass
            else:
                case_lines[case_name][1] = i + 1
                state = 0
        else:
            raise ValueError(f"unexpected state {state}.")

    good_cases = {}
    for cn, d in data.get("cases", {}).items():
        if cn not in case_lines:
            continue

        code = "".join(lines[case_lines[cn][0] : case_lines[cn][1]])
        args, kwargs, invoking_range = extract_params_from_invoking(api, code)
        # if d.get('all_kwargs', False) is True:
        #     good_casenames.append(n)
        # 我只需要位置参数 + 关键字参数总数对就行，
        # 不需要符合全部关键字参数的需求，这样我更容易自动化修复
        # if d.get("all_*args", False) is True or d.get("all_kwargs", False) is True:
        #     good_cases[n] = d
        #     continue

        kwargs_dict = dict(kwargs)
        if len(args) > 0:
            keyed_pargs = zip(data["position_args_list"], args)
            for k, v in keyed_pargs:
                assert k not in kwargs_dict, f"duplicated key {k} in {cn}."
                kwargs_dict[k] = v

        wanted_args = set(
            [k for k in data["keyword_args_list"] if k not in kwargs_dict]
        )

        # 有些参数可以我预先指定默认值啊
        if "device" in wanted_args:
            wanted_args.remove("device")
            kwargs_dict["device"] = "'cpu'"
        if "dtype" in wanted_args:
            wanted_args.remove("dtype")
            kwargs_dict["dtype"] = "torch.float32"

        if len(kwargs_dict) == len(data["keyword_args_list"]):
            good_cases[cn] = kwargs_dict

    if len(good_cases) == 0:
        print(
            f'api {api} in "{file_path}" not has no good cases as template, skip auto fix.'
        )
        return

    append_cases = []
    case_append = lambda c, src: append_cases.append({"code": c, "source": src})

    for cn, kwargs_dict in good_cases.items():
        code = "".join(lines[case_lines[cn][0] : case_lines[cn][1]])
        args, kwargs, invoking_range = extract_params_from_invoking(api, code)

        new_case_template = (
            f'{code[:invoking_range[0]]}({"{}"}){code[invoking_range[1]:]}'
        )

        if data.get("all args", False) is False:
            params = [kwargs_dict[k] for k in data["position_args_list"]]
            case_append(new_case_template.format(", ".join(params)), cn)

        if data.get("all kwargs", False) is False:
            params = [f"{k}={kwargs_dict[k]}" for k in data["keyword_args_list"]]
            case_append(new_case_template.format(", ".join(params)), cn)

        if (
            data.get("kwargs out of order", False) is False
            and len(data["keyword_args_list"]) > 1
        ):
            # 好吧，乱序我打算直接逆序
            params = [f"{k}={kwargs_dict[k]}" for k in data["keyword_args_list"][::-1]]
            case_append(new_case_template.format(", ".join(params)), cn)

        if (
            data.get("all default", False) is False
            and data.get("min_input_args", -1) >= 0
        ):
            params = [
                kwargs_dict[k]
                for k in data["position_args_list"][: data["min_input_args"]]
            ]
            case_append(new_case_template.format(", ".join(params)), cn)

    if len(append_cases) > 0:
        print("##############################")
        print("#  AutoFix")
        print("#")
        print(f'#  file: "{file_path}"')
        print(f"#  {len(append_cases)} cases fixed.")
        print("#")
        print("#  please rerun validate tool.")
        print("##############################")

        with open(file_path, "a+") as f:
            for ncdata in append_cases:
                ac, src = ncdata["code"], ncdata["source"]
                acl = [l.rstrip() + "\n" for l in ac.split("\n")]
                while acl[-1] == "\n":
                    acl.pop()

                assert TESTCASE_NAME_PATTERN.match(acl[0])
                case_id = case_id + 1

                acl[0] = f"def test_case_{case_id}():\n"
                acl[-1] = acl[-1].rstrip("\n")
                acl = (
                    [
                        "\n",
                        "\n",
                        f"# generated by validate_unittest autofix, based on {src}\n",
                    ]
                    + acl
                    + ["\n"]
                )

                f.writelines(acl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call Variety Check v0.2")
    parser.add_argument(
        "--rerun", "-r", dest="files_or_dirs", nargs="+", help="Rerun tests"
    )
    parser.add_argument(
        "--no-check", action="store_true", default=False, help="Disable check"
    )
    parser.add_argument(
        "--report", action="store_true", default=True, help="Generate report"
    )
    parser.add_argument(
        "--richtext",
        action="store_true",
        default=False,
        help="Generate report in richtext format",
    )
    parser.add_argument(
        "--autofix",
        action="store_true",
        default=False,
        help="Auto fix the test file when only 1 file is specified.",
    )

    args = parser.parse_args()

    with open(os.path.join(project_dir, "paconvert/api_mapping.json"), "r") as f:
        api_mapping = json.load(f)
    with open(os.path.join(project_dir, "paconvert/attribute_mapping.json"), "r") as f:
        attribute_mapping = json.load(f)
    with open(os.path.join(project_dir, "paconvert/api_alias_mapping.json"), "r") as f:
        api_alias_mapping = json.load(f)

    test_data_path = os.path.join(output_dir, "validation.json")

    if os.path.exists(test_data_path) and os.path.isfile(test_data_path):
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
    else:
        test_data = {}

    if args.files_or_dirs is not None and len(args.files_or_dirs) > 0:
        newtest_data = get_test_cases(args.files_or_dirs)

        test_attribute_count = 0
        for attribute in attribute_mapping:
            if attribute in newtest_data:
                newtest_data.pop(attribute)
                test_attribute_count += 1
        if test_attribute_count > 0:
            print(f"INFO: {test_attribute_count} attribute unittests are removed.")

        test_data.update(newtest_data)
        with open(test_data_path, "w") as f:
            json.dump(test_data, f, indent=2)

    missing_unittest_list = []
    for api, data in api_mapping.items():
        if "Matcher" not in data:
            continue
        if api not in test_data:
            missing_unittest_list.append(api)
    with open(os.path.join(output_dir, "missing_unittest_list.log"), "w") as f:
        for api in missing_unittest_list:
            print(api, file=f)

    for alias, target in api_alias_mapping.items():
        if target in api_mapping:
            # api_mapping[alias] = api_mapping[target]
            pass
        else:
            # 不应该单独配置 alias 的 api_mapping.json
            if alias in api_mapping:
                raise ValueError(
                    f"alias {alias} should not appear in api_mapping.json."
                )

            # 没有目标的 alias 不应该有单测
            if alias in test_data:
                raise ValueError(
                    f"alias {alias} is not configured but it's unittest exists."
                )

    if not args.no_check:
        report, aux_detailed_data = check_call_variety(
            test_data,
            api_mapping,
            api_alias=api_alias_mapping,
            verbose=(not args.report),
        )
        sorted_report = dict(sorted(report.items(), key=lambda x: x[0]))

        if args.report:
            report_outpath = os.path.join(output_dir, "validation_report.md")
            with open(report_outpath, "w") as f:
                f.write("# Unittest Validation Report\n\n")

                columns = [
                    "api",
                    "all args",
                    "all kwargs",
                    "kwargs out of order",
                    "all default",
                ]
                f.write(f'| {" | ".join(columns)} |\n')
                f.write(f'| {" | ".join(["---"] * len(columns))} |\n')

                item2desc_dict = {
                    True: "✅",
                    False: "❌",
                    None: "⚠️",
                }

                for api, data in sorted_report.items():
                    api_title = api
                    if args.richtext:
                        api_doc_url = simple_map_api_url(
                            api, api_alias=api_alias_mapping
                        )
                        api_title = f"[{api}]({api_doc_url})"

                    if data.get("partial support", False):
                        api_title = f"❓ {api_title}"
                    if data.get("overloadable", False):
                        api_title = f"🔁 {api_title}"
                    if data.get("corner case", False):
                        api_title = f"🟢 {api_title}"

                    f.write(
                        f'| {api_title} | {" | ".join([item2desc_dict[data[k]] for k in columns[1:]])} |\n'
                    )

        if (
            args.autofix
            and args.files_or_dirs is not None
            and len(args.files_or_dirs) == 1
        ):
            file_path = args.files_or_dirs[0]

            selected_aux_data = dict(
                [(api, aux_detailed_data.get(api, {})) for api in newtest_data]
            )
            autofix_single_api(file_path, selected_aux_data)
