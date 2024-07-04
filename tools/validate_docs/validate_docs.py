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

project_dir = os.path.join(os.path.dirname(__file__), "../..")
tool_dir = os.path.dirname(__file__)

context_verbose_level = 1


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

    torch_api = doc_item["torch_api"]
    paddle_api = doc_item["paddle_api"]

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

    if mapped_api != paddle_api:
        raise ValidateError(
            f"{torch_api}: `paddle_api` with UnchangeMatcher is not equal: {mapped_api} != {paddle_api}"
        )


def check_api_mapping(paconvert_item, doc_item):
    matcher = paconvert_item["Matcher"]
    torch_api = doc_item["torch_api"]
    mapping_type = doc_item["mapping_type"]

    mapping_type_1 = [
        "无参数",
        "参数完全一致",
        "仅 paddle 参数更多",
        "仅参数名不一致",
        "仅参数默认值不一致",
    ]

    if mapping_type in mapping_type_1:
        if "paddle_api" not in doc_item:
            raise DocDataError(
                f"{torch_api}: `paddle_api` is not in doc_item: {doc_item}"
            )

        # 不用检查的特例
        if matcher == "UnchangeMatcher":
            return check_unchange_matcher(paconvert_item, doc_item)

        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `paddle_api` is {doc_item['paddle_api']}"
            )
        if doc_item["paddle_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["paddle_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
            )
        return

    mapping_type_2 = ["torch 参数更多"]
    if mapping_type in mapping_type_2:
        if "paddle_api" not in doc_item:
            raise DocDataError(
                f"{torch_api}: `paddle_api` is not in doc_item: {doc_item}"
            )

        # 不用检查的特例
        if matcher == "UnchangeMatcher":
            return check_unchange_matcher(paconvert_item, doc_item)

        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `paddle_api` is {doc_item['paddle_api']}"
            )
        if doc_item["paddle_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["paddle_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
            )
        return

    mapping_type_3 = [
        "返回参数类型不一致",
        "输入参数用法不一致",
        "输入参数类型不一致",
    ]
    if mapping_type in mapping_type_3:
        if "paddle_api" not in doc_item:
            raise DocDataError(
                f"{torch_api}: `paddle_api` is not in doc_item: {doc_item}"
            )

        # 不用检查的特例
        if matcher == "UnchangeMatcher":
            return check_unchange_matcher(paconvert_item, doc_item)

        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `paddle_api` is {doc_item['paddle_api']}"
            )
        if doc_item["paddle_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["paddle_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
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
    parser = argparse.ArgumentParser(description="Call Variety Check v0.1")
    parser.add_argument(
        "--verbose_level",
        type=int,
        default=1,
        help="Specify the verbose level, 0 silent, 1 necessary, 2 info, 3 debug.",
    )
    args = parser.parse_args()

    context_verbose_level = args.verbose_level

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

    with open(os.path.join(tool_dir, "docs_mappings.json"), "r") as f:
        docs_mapping_data = json.load(f)
        docs_mapping = dict([(i["torch_api"], i) for i in docs_mapping_data])

    missing_docs = []
    validated_apis = []

    for api in api_mapping:
        if len(api_mapping[api]) == 0:
            continue
        if "Matcher" not in api_mapping[api]:
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
        validate_errors = []

        for api, docs_api in validated_apis:
            try:
                check_api_mapping(api_mapping[api], docs_mapping[docs_api])
            except DocDataError as e:
                doc_errors.append(e)
            except PaConvertDataError as e:
                paconvert_errors.append(e)
            except ValidateError as e:
                validate_errors.append(e)
            except NotImplementedError as e:
                validate_errors.append(e)
            except Exception as e:
                verbose_print(f"ERROR: {api} raised {e}")
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
        validate_errors.sort(key=lambda e: f"{type(e)}", reverse=True)
        if len(validate_errors) > 0:
            verbose_print(f"ERROR: {len(validate_errors)} api validate error.")
            with open(os.path.join(tool_dir, "validate_error_list.log"), "w") as f:
                for ve in validate_errors:
                    print(ve, file=f)
                    verbose_print(f"INFO: {ve}", v_level=3)
