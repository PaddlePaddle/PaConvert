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

sys.path.append(os.path.dirname(__file__) + "/../..")
from tests.code_library.model_case import MODEL_LIST

RED = "\033[91m"
RESET = "\033[0m"


def process(file):
    return "/".join(file.split("/")[9:])


def convert_pytorch_code_to_paddle():
    convert_fail_list = []
    for pytorch_dir in MODEL_LIST:
        convert_paddle_dir = pytorch_dir.replace("torch_code", "convert_paddle_code")
        exit_code = os.system(
            f"python paconvert/main.py --in_dir {pytorch_dir} --out_dir {convert_paddle_dir}  --log_level 'DEBUG' "
        )
        if exit_code != 0:
            print(f"The {pytorch_dir} convert fail!")
            convert_fail_list.append(pytorch_dir)

    return convert_fail_list


def run_model():
    run_fail_list = []
    for pytorch_dir in MODEL_LIST:
        convert_paddle_dir = pytorch_dir.replace("torch_code", "convert_paddle_code")
        exit_code = os.system(f"python {convert_paddle_dir}")
        if exit_code != 0:
            print(f"{pytorch_dir} -> {convert_paddle_dir} run fail!")
            run_fail_list.append(pytorch_dir)
        else:
            print(f"{pytorch_dir} -> {convert_paddle_dir} run success!")
    return run_fail_list


if __name__ == "__main__":
    convert_fail_list = convert_pytorch_code_to_paddle()
    if convert_fail_list:
        print("*****************************************************************")
        print(RED + "The following pytorch model convert fail!" + RESET)
        for file_dir in convert_fail_list:
            print(f" {file_dir} convert fail!")
        print("*****************************************************************")

    run_fail_list = run_model()
    if run_fail_list:
        print("*****************************************************************")
        print(
            RED + "The following pytorch model convert to paddle but run fail!" + RESET
        )
        for file_dir in run_fail_list:
            print(f" {file_dir} model test " + RED + "fail" + RESET)

        print("******************************************************************")

    if convert_fail_list or run_fail_list:
        sys.exit(1)
