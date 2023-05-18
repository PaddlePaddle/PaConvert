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
from typing import Tuple

sys.path.append(os.path.dirname(__file__) + "/../..")
from tests.code_library.code_case_v1.file_mapping_dict import (
    global_file_mapping_dict,
)


def translate_pytorch_code_to_paddle_code() -> Tuple[bool, list[str]]:
    translate_file_flag = False
    translate_file_fail_list = []
    for pytorch_file, paddle_file in global_file_mapping_dict.items():
        paddle_file = paddle_file.replace("paddle_code", "temp_paddle_code")
        exit_code = os.system(
            f"python paconvert/main.py --in_dir {pytorch_file} --out_dir {paddle_file}"
        )
        if exit_code != 0:
            print(f"The {pytorch_file} translation fail!")
            translate_file_fail_list.append(pytorch_file)
            translate_file_flag = True

    return translate_file_flag, translate_file_fail_list


def compare_file_func(file1, file2) -> bool:
    with open(file1, "r") as f1, open(file2, "r") as f2:
        content1 = f1.read()
        content2 = f2.read()
    return content1 == content2


def compare_file_consistency() -> Tuple[bool, list[str]]:
    file_consistency_flag = False
    file_fail_list = []
    for pytorch_file, paddle_file in global_file_mapping_dict.items():
        temp_paddle_file = paddle_file.replace("paddle_code", "temp_paddle_code")
        if not compare_file_func(paddle_file, temp_paddle_file):
            file_consistency_flag = True
            file_fail_list.append(pytorch_file)
    return file_consistency_flag, file_fail_list


def translation_summary_log(file_list) -> None:
    for file in file_list:
        API = file.split("_")[1:].join(".")
        print(API + " API in " + file + " tranlate fail!")


def file_consistency_summary_log(file_list) -> None:
    for file in file_list:
        pytorch_file_name = file.split("/")[-1]
        paddle_file_name = global_file_mapping_dict[file].split("/")[-1]
        print(
            f"the {pytorch_file_name} translation results and {paddle_file_name} are inconsistent!"
        )


if __name__ == "__main__":
    (
        translate_file_flag,
        translate_file_fail_list,
    ) = translate_pytorch_code_to_paddle_code()
    file_consistency_flag, file_consistency_fail_list = compare_file_consistency()
    if translate_file_flag:
        print(
            "************************************************************************************"
        )
        print("The following pytorch file test case translation fail!")
        translation_summary_log(translate_file_fail_list)
        print(
            "************************************************************************************"
        )

    if file_consistency_flag:
        print(
            "************************************************************************************"
        )
        print("The following pytorch file test case translation inconsistency!")
        file_consistency_summary_log(file_consistency_fail_list)
        print(
            "************************************************************************************"
        )

    if translate_file_flag or file_consistency_flag:
        sys.exit(1)
