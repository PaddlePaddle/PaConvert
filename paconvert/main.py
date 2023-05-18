# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__) + "/..")

from paconvert.converter import Converter
from paconvert.transformer.basic_transformer import change_torch_package_list


def main():
    if sys.version_info < (3, 8):
        raise RuntimeError(
            "PaConvert use new AST syntax and only supports Python version >= 3.8 now."
        )

    parser = argparse.ArgumentParser(
        prog="paconvert", description="PaConverter tool entry point"
    )
    parser.add_argument(
        "--in_dir",
        default="./tests/test_transpose.py",
        type=str,
        help="the input PyTorch file or directory.",
    )
    parser.add_argument(
        "--out_dir", default=None, type=str, help="the output Paddle directory."
    )
    parser.add_argument(
        "--exclude_dirs",
        default=None,
        type=str,
        help="the exclude Pytorch file or directory, which will not be converted.",
    )
    parser.add_argument(
        "--log_dir", default=None, type=str, help="the input PyTorch file or directory."
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO"],
        help="set log level, default is INFO",
    )
    parser.add_argument(
        "--run_check",
        default=False,
        type=bool,
        help="run check the paddle convert tool",
    )
    parser.add_argument(
        "--show_unsupport",
        default=False,
        type=bool,
        help="show these APIs which are not supported to convert now",
    )
    parser.add_argument(
        "--separate_convert",
        default=False,
        type=bool,
        help="Convert Pytorch project each element Separately",
    )

    args = parser.parse_args()

    if args.run_check:
        cwd = os.path.dirname(__file__)
        coverter = Converter(args.log_dir, args.log_level, args.show_unsupport)
        coverter.run(cwd + "/example_code.py", cwd + "/temp_out/example_code.py")
        sys.exit(0)

    if args.separate_convert:
        change_torch_package_list()
        project_num_100 = 0
        project_num_95 = 0
        project_num_90 = 0

        in_dir = os.path.abspath(args.in_dir)
        for pytorch_project in os.listdir(in_dir):
            pytorch_project = os.path.join(in_dir, pytorch_project)
            coverter = Converter(args.log_dir, args.log_level, args.show_unsupport)
            coverter.run(pytorch_project, args.out_dir, args.exclude_dirs)
            if coverter.convert_rate == 1.0:
                project_num_100 += 1
            if coverter.convert_rate >= 0.95:
                project_num_95 += 1
            if coverter.convert_rate >= 0.90:
                project_num_90 += 1

        project_num = len(os.listdir(in_dir))
        print("\n**************************************************************")
        print("Model Set Convert Summary:")
        print("**************************************************************")
        print(
            "There are {} Pytorch Projects:\n : {}({:.2%}) Project's Convert-Rate is 100%\n "
            ": {}({:.2%}) Project's Convert-Rate >=95%\n : {}({:.2%}) Project's Convert-Rate >=90%".format(
                project_num,
                project_num_100,
                project_num_100 / project_num,
                project_num_95,
                project_num_95 / project_num,
                project_num_90,
                project_num_90 / project_num,
            )
        )
        sys.exit(0)

    assert args.in_dir is not None, "User must specify --in_dir "
    coverter = Converter(args.log_dir, args.log_level, args.show_unsupport)
    coverter.run(args.in_dir, args.out_dir, args.exclude_dirs)

    print(r"****************************************************************")
    print(r"______                                   _   ")
    print(r"| ___ \                                 | |  ")
    print(r"| |_/ /_ _  ___ ___  _ ____   _____ _ __| |_ ")
    print(r"|  __/ _  |/ __/ _ \| \_ \ \ / / _ \ \__| __|")
    print(r"| | | (_| | (_| (_) | | | \ V /  __/ |  | |_ ")
    print(r"\_|  \__,_|\___\___/|_| |_|\_/ \___|_|   \__|")
    print(r"")
    print(r"***************************************************************")


if __name__ == "__main__":
    main()
