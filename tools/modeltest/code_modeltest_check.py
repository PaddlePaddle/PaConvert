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

from tests.code_library.model_case.file_mapping_dict import (
    global_file_mapping_dict,
)

RED = "\033[91m"
RESET = "\033[0m"


def process(file):
    return "/".join(file.split("/")[9:])


def translate_pytorch_code_to_paddle_code() -> Tuple[bool, list[str]]:
    translate_file_flag = False
    translate_file_fail_list = []
    for pytorch_file, paddle_file in global_file_mapping_dict.items():
        exit_code = os.system(
            f"python paconvert/main.py --in_dir {pytorch_file} --out_dir {paddle_file}"
        )
        if exit_code != 0:
            print(f"The {process(pytorch_file)} translation fail!")
            translate_file_fail_list.append(pytorch_file)
            translate_file_flag = True

    return translate_file_flag, translate_file_fail_list


def translation_summary_log(file_list) -> None:
    for file in file_list:
        print(f"{process(file)} tranlate fail!")


def test_run() -> Tuple[bool, list[str]]:
    run_file_flag = False
    run_file_fail_list = []
    for pytorch_file, paddle_file in global_file_mapping_dict.items():
        exit_code = os.system(f"python {paddle_file}")
        if exit_code != 0:
            print(f"{process(pytorch_file)} -> {process(paddle_file)} :fail!")
            run_file_fail_list.append(pytorch_file)
            run_file_flag = True

    return run_file_flag, run_file_fail_list


def file_run_summary_log(file_list) -> None:
    for file in file_list:
        print(f"{process(file)} model test " + RED + "fail" + RESET)


if __name__ == "__main__":
    (
        translate_file_flag,
        translate_file_fail_list,
    ) = translate_pytorch_code_to_paddle_code()
    if translate_file_flag:
        print(
            "************************************************************************************"
        )
        print(RED + "The following pytorch file test case translation fail!" + RESET)
        translation_summary_log(translate_file_fail_list)
        print(
            "************************************************************************************"
        )

    run_file_flag, run_file_list = test_run()

    if run_file_flag:
        print(
            "************************************************************************************"
        )
        print(
            RED
            + "The following test case translation paddle file test case run fail!"
            + RESET
        )
        file_run_summary_log(run_file_list)
        print(
            "************************************************************************************"
        )

    if translate_file_flag or run_file_flag:
        sys.exit(1)
