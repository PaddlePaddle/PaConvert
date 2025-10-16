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
from paconvert.global_var import GlobalManager

try:
    from paconvert.version import __version__
except:
    __version__ = "0.0.0"


def main():
    if sys.version_info < (3, 8):
        raise RuntimeError(
            "PaConvert use new AST syntax and only supports Python version >= 3.8 now."
        )

    parser = argparse.ArgumentParser(
        prog="paconvert", description="PaConverter tool entry point"
    )
    parser.add_argument("-V", "--version", action="version", version=f"{__version__}")
    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        help="The input PyTorch file or directory.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=None,
        type=str,
        help="Optional. The output Paddle file or directory. Default: './paddle_project'.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        default=None,
        type=str,
        help="Optional. Regular Pattern. PyTorch file or directory matched will not be converted, multiple patterns should be splited by ',' . Default: None.",
    )
    parser.add_argument(
        "--exclude_packages",
        default=None,
        type=str,
        help="Optional. Those PyTorch packages will not be recognized & converted, multiple packages should be split by ',' . Default is None.",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Optional. The log directory. Default: None.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
        choices=["WARNING", "INFO", "DEBUG"],
        help="Optional. The log level, default is INFO",
    )
    parser.add_argument(
        "--show_all_api",
        action="store_true",
        help="Optional. Show all APIs which should be converted",
    )
    parser.add_argument(
        "--show_unsupport_api",
        action="store_true",
        help="Optional. Show those APIs which are not supported to convert",
    )
    parser.add_argument(
        "--run_check",
        action="store_true",
        help="Optional. Run check the paddle convert tool.",
    )
    parser.add_argument(
        "--no_format",
        action="store_true",
        help="Optional. Disable format the converted code automatically.",
    )
    parser.add_argument(
        "--log_markdown",
        action="store_true",
        help="Inner Usage. Show log with markdown format, user do not need to pay attention.",
    )
    parser.add_argument(
        "--separate_convert",
        action="store_true",
        help="Inner Usage. Convert Pytorch project each element separately, user do not need to pay attention.",
    )
    parser.add_argument(
        "--calculate_speed",
        action="store_true",
        help="Inner Usage. Calculate convert speed. user do not need to pay attention.",
    )
    parser.add_argument(
        "--only_complete",
        action="store_true",
        help="Inner Usage. Complete PyTorch code snippets only. user do not need to pay attention.",
    )
    args = parser.parse_args()

    if args.exclude_packages:
        exclude_packages = args.exclude_packages.split(",")
        for package in exclude_packages:
            GlobalManager.TORCH_PACKAGE_MAPPING.pop(package)

    if args.separate_convert:
        project_num_100 = 0
        project_num_95 = 0
        project_num_90 = 0
        convert_rate_map = {}

        in_dir = os.path.abspath(args.in_dir)
        for project_name in os.listdir(in_dir):
            converter = Converter(
                log_dir=args.log_dir,
                log_level=args.log_level,
                log_markdown=args.log_markdown,
                show_all_api=args.show_all_api,
                show_unsupport_api=args.show_unsupport_api,
                no_format=args.no_format,
                calculate_speed=args.calculate_speed,
                only_complete=args.only_complete,
            )

            project_dir = os.path.join(in_dir, project_name)
            converter.run(project_dir, args.out_dir, args.exclude)
            if converter.convert_rate == 1.0:
                project_num_100 += 1
            if converter.convert_rate >= 0.95:
                project_num_95 += 1
            if converter.convert_rate >= 0.90:
                project_num_90 += 1
            convert_rate_map[project_name] = converter.convert_rate

        project_num = len(os.listdir(in_dir))
        print("\n==============================================")
        print("Convert Summary")
        print("==============================================")
        print("Convert rate of each project is:\n")
        for k, v in convert_rate_map.items():
            print("  {}: {:.2%}".format(k, v))
        print(
            "\nIn Total, there are {} Pytorch Projects:\n : {}({:.2%}) Project's Convert-Rate is 100%\n "
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
        import pandas

        df = pandas.DataFrame.from_dict(dict(convert_rate_map), orient="index")
        df.to_excel("convert_rate_map.xlsx")
        return

    converter = Converter(
        log_dir=args.log_dir,
        log_level=args.log_level,
        log_markdown=args.log_markdown,
        show_all_api=args.show_all_api,
        show_unsupport_api=args.show_unsupport_api,
        no_format=args.no_format,
        calculate_speed=args.calculate_speed,
        only_complete=args.only_complete,
    )

    if args.run_check:
        cwd = os.path.dirname(__file__)
        converter.run(cwd + "/example_code.py", cwd + "/temp_out/example_code.py")
        return

    assert args.in_dir is not None, "User must specify --in_dir "
    converter.run(args.in_dir, args.out_dir, args.exclude)

    print(r"****************************************************************")
    print(r"______      _____                          _   ")
    print(r"| ___ \    / ____|                        | |  ")
    print(r"| |_/ /_ _| |     ___  _ ____   _____ _ __| |_ ")
    print(r"|  __/ _  | |    / _ \| \_ \ \ / / _ \ \__| __|")
    print(r"| | | (_| | |___| (_) | | | \ V /  __/ |  | |_ ")
    print(r"\_|  \__,_|\_____\___/|_| |_|\_/ \___|_|   \__|")
    print(r"")
    print(r"***************************************************************")


if __name__ == "__main__":
    main()
