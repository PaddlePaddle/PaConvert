# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

project_dir = os.path.join(os.path.dirname(__file__), "../..")
tool_dir = os.path.dirname(__file__)

context_verbose_level = 1

validate_whitelist = [
    r"^torch\.(cuda\.)?(\w*)Tensor$",
]


def verbose_print(*args, v_level=1, **kwargs):
    if context_verbose_level >= v_level:
        print(*args, **kwargs)


class DocDataError(Exception):
    pass


class PaConvertDataError(Exception):
    pass


class ValidateError(Exception):
    pass


def check_unchange_matcher(paconvert_item, doc_item):
    matcher = paconvert_item["Matcher"]

    assert matcher == "UnchangeMatcher"

    torch_api = doc_item["src_api"]
    paddle_api = doc_item["dst_api"]

    api_mapping_rules = {
        "torch.optim.": "paddle.optimizer.",
        "torch.nn.Module.": "paddle.nn.Layer.",
        "torch.autograd.function.FunctionCtx.": "paddle.autograd.PyLayerContext.",
        "torch.": "paddle.",
    }

    rules_key = sorted(api_mapping_rules.keys(), key=lambda k: len(k), reverse=True)
    mapped_api = torch_api
    for key in rules_key:
        mapped_api = re.sub(f"^{re.escape(key)}", api_mapping_rules[key], mapped_api)
    if "paddle_api" in paconvert_item:
        mapped_api = paconvert_item.get("paddle_api")

    if mapped_api != paddle_api:
        raise ValidateError(
            f"{torch_api}: `paddle_api` with UnchangeMatcher is not equal: {mapped_api} != {paddle_api}"
        )


DOC_ARG_PATTERN = re.compile(
    r"<\s*font[^>]*\s*>(?P<arg_name>.*?)<\s*/\s*font\s*>", re.IGNORECASE
)


def extract_doc_arg(arg_str, remove_star=True):
    arg_name = arg_str
    m = DOC_ARG_PATTERN.match(arg_name)

    if m:
        arg_name = m["arg_name"].strip()
    else:
        pass

    # 支持类型标注
    if ":" in arg_name:
        arg_name = arg_name.split(":")[0]

    arg_name = arg_name.strip()

    if remove_star and arg_name != "*":
        arg_name = arg_name.lstrip("*")

    return arg_name


def get_kwargs_mapping_from_doc(doc_item):
    args_mapping = doc_item.get("args_mapping", [])
    kwargs_change = {}

    for am in args_mapping:
        at = extract_doc_arg(am["src_arg"])
        ap = extract_doc_arg(am["dst_arg"])
        note = am["note"]

        if at == "-":
            continue
        elif ap == "-":
            continue
        elif at == "返回值":
            continue
        elif "," in at:
            continue
        elif "," in ap:
            continue
        else:
            kwargs_change[at] = ap

    return kwargs_change


# 如果参数映射在这个里面，则忽略检查，因为不是对应关系
IGNORE_KWARGS_CHANGE_PAIRS = {
    ("self", "x"),
    ("some", "mode"),
    ("non_blocking", "blocking"),
    ("requires_grad", "stop_gradient"),
    ("track_running_stats", "use_global_stats"),
    ("async_op", "sync_op"),
    ("time_major", "batch_first"),
}


# 如果参数映射在这个里面，则进行参数映射的转换
KWARGS_CHANGE_CHANGE_DICT = {
    # for *split
    "split_size_or_sections:num_or_indices": {
        "indices": "num_or_indices",
        "sections": "num_or_indices",
    },
    "indices_or_sections:num_or_indices": {
        "indices": "num_or_indices",
        "sections": "num_or_indices",
    },
}


PRESET_MATCHER_KWARGS_CHANGE_PAIRS = {
    "CreateMatcher": {"size": "shape"},
    "Num2TensorBinaryMatcher": {"input": "x", "other": "y"},
    "DivideMatcher": {"input": "x", "other": "y"},
    "IndexAddMatcher": {"source": "value"},
    "IInfoMatcher": {"type": "dtype"},
    "ZeroGradMatcher": {"set_to_none": "set_to_zero"},
    "SvdMatcher": {"some": "full_matrics"},
    "AtleastMatcher": {"tensors": "inputs"},
    "SLogDetMatcher": {"A": "x", "input": "x"},
    "MeshgridMatcher": {"tensors": "args"},
    "RNNBaseMatcher": {"batch_first": "time_major", "bidirectional": "direction"},
    "AvgPoolMatcher": {"input": "x", "count_include_pad": "exclusive"},
    "RoundMatcher": {"input": "x"},
    "FunctionalPadMatcher": {"input": "x"},
    "FunctionalSmoothL1LossMatcher": {"beta": "delta"},
    "OptimOptimizerMatcher": {"params": "parameters"},
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
    "torch.sum",
    "torch.nansum",
    "torch.linalg.matrix_rank",
    # *split kwargs is processed through KWARGS_CHANGE_CHANGE_DICT,
    # but overload `args_list` is not supported, so ignore it.
    # (int sections) or (tuple of ints indices)
    "torch.Tensor.dsplit",
    "torch.Tensor.hsplit",
    "torch.Tensor.tensor_split",
    "torch.dsplit",
    "torch.hsplit",
    "torch.tensor_split",
}

cornercase_api_aux_dict = {
    "torch.Tensor.type": "torch.Tensor.type with TensorTypeMatcher need support torch.nn.Module.type and torch.nn.Module.type.",
    "torch.Tensor.triangular_solve": "torch.Tensor.triangular_solve with TriangularSolveMatcher is too complex.",
    "torch.cuda.nvtx.range_push": "paddle api only support position args, so kwargs_change not works.",
    "torch.utils.cpp_extension.CUDAExtension": "torch.utils.cpp_extension.CUDAExtension with CUDAExtensionMatcher list some kwargs.",
    "torch.utils.cpp_extension.CppExtension": "torch.utils.cpp_extension.CppExtension with CUDAExtensionMatcher list some kwargs.",
    "torch.utils.cpp_extension.BuildExtension": "torch.utils.cpp_extension.BuildExtension only list some kwargs.",
    "torch.nn.Sequential": "var_arg `arg` is processed by SequentialMatcher.",
    # bad case, need fix
    "torch.cuda.stream": "`paddle.device.cuda.stream_guard` args is not as same as `paddle.device.stream_guard`",
    "torch.nn.Module.named_buffers": "remove_duplicate arg is not supported in paddle.",
    "torch.nn.Module.named_modules": "remove_duplicate arg is not supported in paddle.",
    "torch.nn.Module.named_parameters": "remove_duplicate arg is not supported in paddle.",
}


def check_mapping_args(paconvert_item, doc_item):
    if doc_item["mapping_type"] == "组合替代实现":
        return

    torch_api = doc_item["src_api"]

    matcher = paconvert_item["Matcher"]

    args_list = [
        extract_doc_arg(a["arg_name"], remove_star=False)
        for a in doc_item["src_signature"].get("args", [])
    ]
    if args_list == []:
        assert (
            len(paconvert_item.get("args_list", [])) == 0
        ), "`args_list` should not be in paconvert_item."
        # assert 'args_mapping' not in doc_item, f'`args_mapping` should not be in doc_item.'

    # compare kwargs_change from doc and api_mapping.json
    kwargs_change = get_kwargs_mapping_from_doc(doc_item)

    pc_kwargs_change = paconvert_item.get("kwargs_change", {})
    preset_kwargs_change = PRESET_MATCHER_KWARGS_CHANGE_PAIRS.get(matcher, {})
    for k, v in preset_kwargs_change.items():
        if k in pc_kwargs_change:
            continue
        pc_kwargs_change[k] = v

    # 用副本作为检查来源，避免 inplace 修改出现问题
    for k, v in kwargs_change.copy().items():
        index = f"{k}:{v}"

        if index in KWARGS_CHANGE_CHANGE_DICT:
            kwargs_change.pop(k)
            for new_k, new_v in KWARGS_CHANGE_CHANGE_DICT[index].items():
                # 如果有设置同名项，就不更新了
                if new_k not in kwargs_change:
                    kwargs_change[new_k] = new_v

    kwargs_change_equal = True
    for k, v in kwargs_change.items():
        if (k, v) in IGNORE_KWARGS_CHANGE_PAIRS:
            continue
        if pc_kwargs_change.get(k, k) != v:
            kwargs_change_equal = False
            break

    if not kwargs_change_equal:
        raise ValidateError(
            f'{doc_item["src_api"]} {matcher}: `kwargs_change` not match: doc is {kwargs_change}, but paconvert is {paconvert_item.get("kwargs_change", {})}'
        )

    pc_args_list = paconvert_item.get("args_list", [])
    for pa in pc_args_list:
        if pa == "*" or pa == "/":
            continue
        if pa not in args_list:
            raise ValidateError(
                f'{doc_item["src_api"]} {matcher}: `args_list` not match: paconvert is {pc_args_list}, but doc is {args_list}'
            )
    for da in args_list:
        if da == "*args" or da == "**kwargs":
            continue
        if da not in pc_args_list:
            raise ValidateError(
                f'{doc_item["src_api"]} {matcher}: `args_list` not match: paconvert is {pc_args_list}, but doc is {args_list}'
            )


def check_api_mapping(paconvert_item, doc_item):
    matcher = paconvert_item["Matcher"]
    torch_api = doc_item["src_api"]
    mapping_type = doc_item["mapping_type"]

    mapping_type_1 = [
        "无参数",
        "参数完全一致",
        "仅参数名不一致",
        "参数默认值不一致",
        "paddle 参数更多",
    ]

    if mapping_type in mapping_type_1:
        if "dst_api" not in doc_item:
            raise DocDataError(f"{torch_api}: `dst` is not in doc_item: {doc_item}")

        # 不用检查的特例
        if matcher == "UnchangeMatcher":
            return check_unchange_matcher(paconvert_item, doc_item)

        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `paddle_api` is {doc_item['paddle_api']}"
            )
        if doc_item["dst_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["dst_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
            )
        return

    mapping_type_2 = ["torch 参数更多"]
    if mapping_type in mapping_type_2:
        if "dst_api" not in doc_item:
            raise DocDataError(f"{torch_api}: `dst_api` is not in doc_item: {doc_item}")

        # 不用检查的特例
        if matcher == "UnchangeMatcher":
            return check_unchange_matcher(paconvert_item, doc_item)

        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `dst_api` is {doc_item['dst_api']}"
            )
        if doc_item["dst_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["dst_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
            )
        return

    mapping_type_3 = [
        "返回参数类型不一致",
        "输入参数用法不一致",
        "输入参数类型不一致",
    ]
    if mapping_type in mapping_type_3:
        if "dst_api" not in doc_item:
            raise DocDataError(f"{torch_api}: `dst_api` is not in doc_item: {doc_item}")

        # 不用检查的特例
        if matcher == "UnchangeMatcher":
            return check_unchange_matcher(paconvert_item, doc_item)

        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `dst_api` is {doc_item['dst_api']}"
            )
        if doc_item["dst_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["dst_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
            )
        return

    mapping_type_4 = ["组合替代实现"]
    if mapping_type in mapping_type_4:
        # TODO: check
        return

    mapping_type_delete = ["可删除"]
    if mapping_type in mapping_type_delete:
        return

    raise NotImplementedError(
        f"{torch_api}: `mapping_type` not found or not implemented: {mapping_type}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docs Bidirectional Check v0.1")
    parser.add_argument(
        "--unittest_validation",
        type=str,
        default=None,
        help="Specify the unittest validation file path.",
    )
    parser.add_argument(
        "--docs_mappings",
        type=str,
        default=os.path.join(tool_dir, "docs_mappings.json"),
        help="Sepcify the docs_mappings.json (from docs/ repo) file path",
    )
    parser.add_argument(
        "--verbose_level",
        type=int,
        default=1,
        help="Specify the verbose level, 0 silent, 1 necessary, 2 info, 3 debug.",
    )
    args = parser.parse_args()

    context_verbose_level = args.verbose_level

    if args.unittest_validation is not None:
        with open(args.unittest_validation, "r") as f:
            unittest_validation_data = json.load(f)
    else:
        unittest_validation_data = None

    with open(os.path.join(project_dir, "paconvert/api_mapping.json"), "r") as f:
        api_mapping = json.load(f)
    with open(os.path.join(project_dir, "paconvert/attribute_mapping.json"), "r") as f:
        attribute_mapping = json.load(f)
    with open(os.path.join(project_dir, "paconvert/api_alias_mapping.json"), "r") as f:
        api_alias_mapping = json.load(f)

        api_alias_backward_mapping = dict()
        for k, v in api_alias_mapping.items():
            if v not in api_alias_backward_mapping:
                api_alias_backward_mapping[v] = [k]
            else:
                api_alias_backward_mapping[v].append(k)
        for k in api_alias_backward_mapping:
            api_alias_backward_mapping[k] = sorted(
                api_alias_backward_mapping[k], key=lambda x: len(x)
            )
        # 允许有多个原 api，但只有一个目标 api

    with open(args.docs_mappings, "r") as f:
        docs_mapping_data = json.load(f)
        docs_mapping = dict([(i["src_api"], i) for i in docs_mapping_data])

    missing_docs = []
    validated_apis = []

    for api in api_mapping:
        if len(api_mapping[api]) == 0:
            continue
        if "Matcher" not in api_mapping[api]:
            continue

        whitelist_skip = False
        for wl in validate_whitelist:
            if re.match(wl, api):
                whitelist_skip = True
                break
        if whitelist_skip:
            continue

        if api in docs_mapping:
            docs_api = api
            # 反查时先直接查 target，找不到再从短到长匹配
        else:
            for n in api_alias_backward_mapping.get(api, []):
                if n in docs_mapping:
                    docs_api = n
                    break
            else:
                missing_docs.append(api)
                continue

        validated_apis.append((api, docs_api))

    if len(missing_docs) > 0:
        verbose_print(
            f"WARNING: {len(missing_docs)} api do not have mapping docs in `PaddlePaddle/docs`."
        )
        with open(os.path.join(tool_dir, "missing_docs_list.log"), "w") as f:
            for md in missing_docs:
                print(md, file=f)
                verbose_print(f"INFO: api `{md}` has no mapping doc.", v_level=3)

    if len(validated_apis) > 0:
        verbose_print(f"INFO: {len(validated_apis)} api will be validate by docs data.")
        doc_errors = []
        paconvert_errors = []
        validate_errors = {}

        for api, docs_api in validated_apis:
            try:
                check_api_mapping(api_mapping[api], docs_mapping[docs_api])
                check_mapping_args(api_mapping[api], docs_mapping[docs_api])
            except DocDataError as e:
                doc_errors.append(e)
            except PaConvertDataError as e:
                paconvert_errors.append(e)
            except ValidateError as e:
                validate_errors[api] = e
            except NotImplementedError as e:
                validate_errors[api] = e
            except Exception as e:
                verbose_print(f"ERROR: {api} raised {e}")
                traceback.print_exc()
                sys.exit(1)

        if len(doc_errors) > 0:
            verbose_print(f"ERROR: {len(doc_errors)} api doc data error.")
            with open(os.path.join(tool_dir, "doc_data_error_list.log"), "w") as f:
                for de in doc_errors:
                    print(de, file=f)
                    verbose_print(f"INFO: {de}", v_level=3)
        if len(paconvert_errors) > 0:
            verbose_print(f"ERROR: {len(paconvert_errors)} api paconvert data error.")
            with open(
                os.path.join(tool_dir, "paconvert_data_error_list.log"), "w"
            ) as f:
                for pe in paconvert_errors:
                    print(pe, file=f)
                    verbose_print(f"INFO: {pe}", v_level=3)

        validate_errors = [(api, e) for api, e in validate_errors.items()]
        validate_errors.sort(key=lambda e: (api, f"{type(e[1])}"))
        # validate_errors.sort(key=lambda e: f"{type(e)}", reverse=True)
        if len(validate_errors) > 0:
            verbose_print(f"ERROR: {len(validate_errors)} api validate error.")
            api_count_to_check = 0
            with open(os.path.join(tool_dir, "validate_error_list.log"), "w") as f:
                for api, ve in validate_errors:
                    if unittest_validation_data is not None:
                        if api not in unittest_validation_data:
                            print("INFO: NO-UNITTEST", ve, file=f)
                            verbose_print(f"INFO: NO-UNITTEST {ve}", v_level=3)
                            continue
                    if api in overloadable_api_aux_set:
                        print("INFO: OVERLOADABLE", ve, file=f)
                        verbose_print(f"INFO: OVERLOADABLE {ve}", v_level=3)
                        continue
                    if api in cornercase_api_aux_dict:
                        print(
                            f"INFO: CORNERCASE {ve}, REASON {cornercase_api_aux_dict[api]}",
                            ve,
                            file=f,
                        )
                        verbose_print(f"INFO: CORNERCASE {ve}", v_level=3)
                        continue

                    api_count_to_check += 1
                    print(f"WARNING: {ve}", file=f)
                    verbose_print(f"WARNING: {ve}", v_level=3)

            if api_count_to_check > 0:
                verbose_print(
                    f"ERROR: {api_count_to_check} api need to be check manually."
                )
            else:
                verbose_print("INFO: ALL validate error api has been checked.")
