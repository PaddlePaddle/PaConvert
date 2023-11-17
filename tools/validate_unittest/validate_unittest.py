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

whitelist_pattern = [
    r"^test_distributed_all_gather_object\.py",  # 分布式，本地没跑完，先跳过
    r"^test_hub_download_url_to_file\.py",  # 要下载，费时间，先跳过了
    r"^test_(\w*)Tensor\.py",  # 特殊类构造函数，api_mapping.json 不合用，跳过
    r"^test_Size\.py",  # api_mapping.json 不合用
    r"^test_nn_ParameterList\.py",  # 该文件没有测试 torch.nn.ParameterLisaa
    r"^test_utils_cpp_extension_BuildExtension\.py",  # 该文件测试的 api 没有调用
    r"^test_utils_data_Sampler\.py",  # 该文件测试时仅将 api 作为基类继承
    r"^test_utils_data_SequentialSampler\.py",  # 该文件测试时仅将 api 作为基类继承
]


def get_test_cases(discovery_paths=["tests"]):
    # Collect the test cases
    monkeypatch = pytest.MonkeyPatch()
    unittest_data = {}
    pytest.main(
        discovery_paths + [], plugins=[RecordAPIRunPlugin(monkeypatch, unittest_data)]
    )
    return unittest_data


class RecordAPIRunPlugin:
    def __init__(self, monkeypatch, data):
        self.monkeypatch = monkeypatch
        self.collected_items = data
        self.whitelist_re = [re.compile(p) for p in whitelist_pattern]
        originapi = apibase.APIBase.run

        def update_record(api, code, unsupport=False):
            sub_title = "code" if not unsupport else "unsupport"
            records = self.collected_items.get(api, {})
            sub_records = records.get(sub_title, [])
            sub_records.append(code)
            records.update({sub_title: sub_records})
            self.collected_items.update({api: records})

        def record_and_run(self, *args, **kwargs):
            pytorch_api = self.pytorch_api
            assert len(args) > 0
            pytorch_code = args[0]
            unsupport = kwargs.get("unsupport", False)
            update_record(pytorch_api, pytorch_code, unsupport)
            return originapi(self, *args, **kwargs)

        monkeypatch.setattr(apibase.APIBase, "run", record_and_run)

    def __del__(self):
        self.monkeypatch.undo()

    def pytest_collection_modifyitems(self, items):
        # Mount the monkeypatch
        collected = []
        for item in items:
            for w_pat in self.whitelist_re:
                if w_pat.match(item.nodeid):
                    break
            else:
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

    code, idx = code[idx + len(api_name) :], 0

    pair_stack = []
    key = None

    # 寻找参数列表的结束位置
    for i, c in enumerate(code):
        if c == "(" or c == "[":
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
        elif c == "," or c == ")" or c == "]":
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
                    break
            elif c == "]":
                assert pair_stack[-1] == "[", f"Unpaired in {repr(code)}"
                pair_stack.pop()
        elif code[i : i + 7] == "lambda ":
            pair_stack.append("lambda")
        elif c == ":" and pair_stack[-1] == "lambda":
            pair_stack.pop()

    if len(pair_stack) != 0:
        raise ValueError(f"Unpaired in {repr(code)}")

    return args, kwargs


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


def check_call_variety(test_data, api_mapping, verbose=True):
    report = {}
    for api, unittest_data in test_data.items():
        if api not in api_mapping:
            if "code" not in unittest_data:
                if verbose:
                    print(f"SKIP: api {api} is not prepared, skip validations.")
            else:
                print(f"WARNING: api {api} not found in api_mapping")
            continue

        mapping_data = api_mapping[api]

        if "code" not in unittest_data:
            "如果 = 0，说明该 api 只是占位符"
            if len(mapping_data) > 0 and "Matcher" in mapping_data:
                print(f"WARNING: api {api} has no unittest.")
            continue

        abstract = mapping_data.get("abstract", False)
        if abstract:
            print(f"SKIP: api {api} is abstract, skip validations.")
            continue

        if "args_list" not in mapping_data:
            print(f"WARNING: {api} has no mapping data 'args_list'.")
            continue

        min_input_args = mapping_data.get("min_input_args", -1)

        args_list_full = mapping_data.get("args_list", [])

        var_arg_name = None
        var_kwarg_name = None

        _args_list_position_end = len(args_list_full)
        args_list_full_keyword = []
        is_token = lambda x: x.isalpha() or x == "_"
        for i, arg in enumerate(args_list_full):
            if arg.startswith("*"):
                _args_list_position_end = min(_args_list_position_end, i)
                if arg.startswith("**"):
                    if len(arg) > 2 and not is_token(arg[2]):
                        raise ValueError(f'api {api} has unexpected arg "{arg}".')
                    # 允许匿名可变参数列表，如 **kwargs 或 **
                    var_kwarg_name = arg[2:]
                else:
                    var_arg_name = arg[1:]
            elif is_token(arg[0]):
                args_list_full_keyword.append(arg)
            else:
                raise ValueError(f'api {api} has unexpected arg "{arg}".')

        args_list_full_positional = args_list_full[:_args_list_position_end]

        # 这里只移除了不支持的参数，但是事实上，移除不支持的参数会影响其他检查，如
        # (a, b, c, d) 中移除了 (c)，那么不指定关键字最多只能传入 a、b
        unsupport_args = mapping_data.get("unsupport_args", [])
        args_list = [arg for arg in args_list_full if arg not in unsupport_args]

        args_list_positional = [
            arg for arg in args_list_full_positional if arg not in unsupport_args
        ]
        args_list_keyword = [
            arg for arg in args_list_full_keyword if arg not in unsupport_args
        ]

        all_args = False
        all_kwargs = False
        not_subsequence = False
        all_default = False if "min_input_args" in mapping_data else None

        for code in unittest_data["code"]:
            try:
                api_name = extract_api_name(api, code)
                args, kwargs = extract_params_from_invoking(api, code)
            except Exception:
                print(f'Error when extract params from invoking "{api}"')
                print(traceback.format_exc())
                exit(-1)

            # 检查 kwargs 是否符合预设的 args_list
            # 条件是，如果没有 **kwargs 参数，那么 kwargs 的 key 必须是 args_list 的子集
            allowed_keys = args_list_keyword.copy()
            if var_arg_name is not None and len(var_arg_name) > 0:
                allowed_keys.append(var_arg_name)
            if api == "torch.max" or api == "torch.min":
                # 这个函数居然有重载，单独特判吧先
                allowed_keys.append("other")

            if var_kwarg_name is None and len(kwargs) > 0:
                for k, v in kwargs:
                    if k not in allowed_keys:
                        raise ValueError(
                            f"{api}(*{args}, **{kwargs}) has unexpected keyword argument '{k}'."
                        )

            # 如果没有 *arg，args 的长度必须小于等于 args_list_positional 的长度
            if var_arg_name is None and len(args) > len(args_list_positional):
                raise ValueError(
                    f"{api}(*{args}, **{kwargs}) has too many position arguments, args_list={args_list_keyword}"
                )

            if all_default is False:
                if len(args) == min_input_args and len(kwargs) == 0:
                    all_default = True

            if len(args) == len(args_list_positional):
                all_args = True

            keys = [k[0].strip() for k in kwargs]
            if len(keys) == len(args_list_keyword) and match_subsequence(
                args_list_keyword, keys
            ):
                all_kwargs = True
            if len(args_list_keyword) <= 1 or not match_subsequence(
                args_list_keyword, keys
            ):
                not_subsequence = True

            # if api == 'torch.Tensor.add':
            #     print(args_list_with_key, keys)
            #     print(all_args, all_kwargs, not_subsequence, code)

        if all_args and all_kwargs and not_subsequence and all_default is True:
            continue

        report[api] = {
            "all args": all_args,
            "all kwargs": all_kwargs,
            "kwargs out of order": not_subsequence,
            "all default": all_default,
        }

        if verbose:
            if not all_args:
                print(
                    f"INFO: {api} has no unittest with all arguments without keyword."
                )
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

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call Variety Check v0.1")
    parser.add_argument(
        "--rerun", "-r", dest="files_or_dirs", nargs="+", help="Rerun tests"
    )
    parser.add_argument(
        "--no-check", action="store_true", default=False, help="Disable check"
    )
    parser.add_argument(
        "--report", action="store_true", default=True, help="Generate report"
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
        test_data.update(newtest_data)
        with open(test_data_path, "w") as f:
            json.dump(test_data, f)

    for alias, target in api_alias_mapping.items():
        if target in api_mapping:
            api_mapping[alias] = api_mapping[target]
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

    test_attribute_count = 0
    for attribute in attribute_mapping:
        if attribute in test_data:
            test_data.pop(attribute)
            test_attribute_count += 1
    print(f"INFO: {test_attribute_count} attribute unittests are removed.")

    if not args.no_check:
        report = check_call_variety(test_data, api_mapping, verbose=(not args.report))
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
                    f.write(
                        f'| {api} | {" | ".join([item2desc_dict[data[k]] for k in columns[1:]])} |\n'
                    )
