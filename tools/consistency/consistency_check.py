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
"""
   isort:skip_file
"""
import os
import sys
import re

from difflib import context_diff

sys.path.append(os.path.dirname(__file__) + "/../..")
from tests.code_library.code_case import CODE_CONSISTENCY_MAPPING


def convert_pytorch_code_to_paddle():
    convert_fail_list = []
    for pytorch_dir, _ in CODE_CONSISTENCY_MAPPING.items():
        convert_paddle_dir = pytorch_dir.replace("torch_code", "convert_paddle_code")
        exit_code = os.system(
            f"python paconvert/main.py --in_dir {pytorch_dir} --out_dir {convert_paddle_dir}"
        )
        if exit_code != 0:
            print(f"The {pytorch_dir} convert fail!")
            convert_fail_list.append(pytorch_dir)

    return convert_fail_list


def _compare_content(actual_dir, expect_dir):
    result = True
    if os.path.isfile(actual_dir):
        assert os.path.isfile(expect_dir), f"{expect_dir} shoule be a file!"
        with open(actual_dir, "r") as f1, open(expect_dir, "r") as f2:
            content1 = f1.read().strip()
            content2 = f2.read().strip()
            # 对随机的辅助代码路径进行处理，使用正则表达式匹配并替换
            pattern = re.compile(r"sys\.path\.append\((.*?)utils\'(.*?)\)", re.DOTALL)
            content1 = re.sub(
                pattern,
                "sys.path.append('/workspace/PaConvert/paddle_project/utils')",
                content1,
            )
            if content1 != content2:
                # print inconsistent content
                diff = context_diff(
                    content2.split("\n"),
                    content1.split("\n"),
                    fromfile=expect_dir,
                    tofile=actual_dir,
                )
                for line in diff:
                    print(line)
                return False
    elif os.path.isdir(actual_dir):
        assert os.path.isdir(expect_dir), f"{expect_dir} shoule be a dir!"
        for item in os.listdir(actual_dir):
            new_actual_dir = os.path.join(actual_dir, item)
            new_expect_dir = os.path.join(expect_dir, item)
            result *= _compare_content(new_actual_dir, new_expect_dir)

    return result


def compare_code_consistency():
    compare_fail_list = []
    for pytorch_dir, paddle_dir in CODE_CONSISTENCY_MAPPING.items():

        pytorch_file_name = pytorch_dir.split("/")[-1]

        convert_paddle_dir = pytorch_dir.replace("torch_code", "convert_paddle_code")
        if not _compare_content(convert_paddle_dir, paddle_dir):
            compare_fail_list.append(pytorch_dir)
    return compare_fail_list


if __name__ == "__main__":
    convert_fail_list = convert_pytorch_code_to_paddle()
    if convert_fail_list:
        print(
            "*************************************************************************"
        )
        print("The following pytorch file convert fail!")
        for file_dir in convert_fail_list:
            print(f" {file_dir} convert fail!")
        print(
            "*************************************************************************"
        )

    compare_fail_list = compare_code_consistency()
    if compare_fail_list:
        print(
            "*************************************************************************"
        )
        print("The following pytorch file convert inconsistency!")
        for file_dir in compare_fail_list:
            paddle_dir = CODE_CONSISTENCY_MAPPING[file_dir]
            print(f" {file_dir} convert result is inconsistent with {paddle_dir}!")
        print(
            "*************************************************************************"
        )

    if compare_fail_list or convert_fail_list:
        sys.exit(1)
