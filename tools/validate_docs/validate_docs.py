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

    if re.sub(r"^torch\.", "paddle.", torch_api) != paddle_api:
        raise ValidateError(
            f"{torch_api}: `paddle_api` is not equal: {torch_api} != {paddle_api}"
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
        if "paddle_api" not in paconvert_item:
            raise PaConvertDataError(
                f"{torch_api}: `paddle_api` is not in paconvert_item: {paconvert_item}, but doc `paddle_api` is {doc_item['paddle_api']}"
            )
        if doc_item["paddle_api"] != paconvert_item["paddle_api"]:
            raise ValidateError(
                f'{torch_api}: `paddle_api` not match: doc is `{doc_item["paddle_api"]}`, but paconvert is `{paconvert_item["paddle_api"]}`'
            )
        return

    else:
        raise NotImplementedError(
            f"{torch_api}: `mapping_type` not found: {mapping_type}"
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

        if api not in docs_mapping:
            missing_docs.append(api)
        else:
            validated_apis.append(api)

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

        for api in validated_apis:
            try:
                check_api_mapping(api_mapping[api], docs_mapping[api])
            except DocDataError as e:
                doc_errors.append(e)
            except PaConvertDataError as e:
                paconvert_errors.append(e)
            except ValidateError as e:
                validate_errors.append(e)
            except NotImplementedError as e:
                validate_errors.append(e)
            except Exception as e:
                verbose_print(f"ERROR: {api} {e}")
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
