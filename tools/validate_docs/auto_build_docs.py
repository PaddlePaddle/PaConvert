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

project_dir = os.path.join(os.path.dirname(__file__), "../..")
tool_dir = os.path.dirname(__file__)

mappingType2mappingDesc = {
    "无参数": "两者功能一致，无参数。",
    "参数完全一致": "功能一致，参数完全一致，具体如下：",
    "torch 参数更多": "PyTorch 相比 Paddle 支持更多其他参数，具体如下：",
    "仅参数名不一致": "其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：",
}

mappingTypeShowArgCompare = {
    "参数完全一致": True,
    "无参数": False,
    "torch 参数更多": True,
}

not_supported_matcher = {
    "SetUpMatcher",
    "TensorMatcher",
    "GeneratorMatcher",
    "SizeMatcher",
    "AllcloseMatcher",
    "ErfCMatcher",
    "TensorIndexCopyMatcher",
    "TensorMaxMinMatcher",
    "TensorNew_Matcher",
    "TensorNewFullMatcher",
    "TensorNewTensorMatcher",
    "TensorRenameMatcher",
    "DeviceMatcher",
}

not_supported_api_list = {
    "torch.distributed.rpc.init_rpc",
}


def markdown_title_escape(s):
    return s.replace("_", r"\_")


def get_mapping_type(api_mapping):
    matcher = api_mapping["Matcher"]
    mapping_type = None

    args_list = api_mapping.get("args_list", [])
    if matcher == "UnchangeMatcher":
        if len(args_list) == 0:
            return "无参数"
        else:
            print(args_list)
            raise ValueError("UnchangeMatcher should not have args_list.")
    elif matcher in {
        "GenericMatcher",
        "Num2TensorBinaryWithAlphaMatcher",
        "Num2TensorBinaryMatcher",
    }:
        if len(args_list) == 0:
            return "无参数"

        kwargs_change = api_mapping.get("kwargs_change", {})
        if len(kwargs_change) == 0:
            return "参数完全一致"

        for k, v in kwargs_change.items():
            if len(v.strip()) == 0:
                return "torch 参数更多"

        return "仅参数名不一致"
    else:
        raise ValueError(f"matcher {matcher} not supported.")

    return mapping_type


def auto_build_mapping_doc(api_name, api_mapping):
    matcher = api_mapping["Matcher"]
    mapping_type = None

    args_list = api_mapping.get("args_list", [])
    mapping_type = get_mapping_type(api_mapping)

    if "paddle_api" not in api_mapping:
        if mapping_type in ["无参数"]:
            api_mapping["paddle_api"] = re.sub(r"^torch\.", "paddle.", api_name)

    torch_api_url = f"https://pytorch.org/docs/stable/generated/{api_name}.html"
    torch_signature = f"{api_name}(" + ", ".join(args_list) + ")"

    doc_content = []

    assert mapping_type is not None
    doc_content.append(f"## [ {mapping_type} ]{api_name}\n")

    doc_content.append(f"### [{markdown_title_escape(api_name)}]({torch_api_url})\n")

    doc_content.append(
        f"""```python
{torch_signature}
```
"""
    )

    if "paddle_api" in api_mapping:
        paddle_api = api_mapping["paddle_api"]
        doc_content.append(
            f"### [{markdown_title_escape(paddle_api)}]((url_placeholder))\n"
        )
        kwargs_change = api_mapping.get("kwargs_change", {})

        paddle_signature = (
            f"{paddle_api}("
            + ", ".join(
                [
                    kwargs_change.get(a, a)
                    for a in args_list
                    if len(kwargs_change.get(a, a)) > 0
                ]
            )
            + ")"
        )

        doc_content.append(
            f"""```python
please check whether signature correct.
{paddle_signature}
```
"""
        )

    assert (
        mapping_type in mappingType2mappingDesc
    ), f"mapping_type {mapping_type} not supported."

    doc_content.append(mappingType2mappingDesc[mapping_type] + "\n")

    if mappingTypeShowArgCompare.get(mapping_type, True):
        doc_content.append("### 参数映射\n")
        kwargs_change = api_mapping.get("kwargs_change", {})
        tables = [["PyTorch", "PaddlePaddle", "备注"]]

        for i in args_list:
            tables.append([i, kwargs_change.get(i, i), ""])

        content_length = [
            max([len(i[j]) for i in tables]) for j in range(len(tables[0]))
        ]

        doc_content.append(
            "| "
            + " | ".join([i.ljust(content_length[j]) for j, i in enumerate(tables[0])])
            + " |"
        )
        doc_content.append("| " + " | ".join(["-" * i for i in content_length]) + " |")
        for i in tables[1:]:
            doc_content.append(
                "| "
                + " | ".join([i[j].ljust(content_length[j]) for j in range(len(i))])
                + " |"
            )

    return doc_content


def get_output_file_path(output_dir, api_name):
    prefix2subdir = {
        "torch.Tensor": "Tensor",
        "torch.distributions": "distributions",
        "torch.fft": "fft",
    }

    for prefix in prefix2subdir:
        if api_name.startswith(prefix):
            subdir = prefix2subdir[prefix]
            break
    else:
        print(f"api_name {api_name} have no preset output subdir path.")
        subdir = "ops"

    return os.path.join(output_dir, subdir, f"{api_name}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Build Docs v0.1")
    parser.add_argument(
        "--output_path",
        type=str,
        help="the api_difference dir path.",
    )
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        raise ValueError("output_path should not be None.")
    if not os.path.exists(output_path):
        raise ValueError(f"output_path {output_path} not exists.")

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

    founded_count = 0
    missing_docs = []

    for api in api_mapping:
        if len(api_mapping[api]) == 0:
            continue
        if "Matcher" not in api_mapping[api]:
            continue

        matcher = api_mapping[api]["Matcher"]
        if matcher in not_supported_matcher:
            continue
        if api in not_supported_api_list:
            continue

        docs_api = api
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
                print(api)

        if docs_api not in docs_mapping:
            doc_content = auto_build_mapping_doc(api, api_mapping[api])
            print(doc_content)
            target_path = get_output_file_path(output_path, api)

            if os.path.exists(target_path):
                raise FileExistsError(f"target_path {target_path} already exists.")

            with open(target_path, "w") as f:
                f.write("\n".join(doc_content))

            print(f'auto build doc for {api} success, output to "{target_path}"')
            exit(0)
        else:
            founded_count += 1

    print(f"found {founded_count} api exists.")
