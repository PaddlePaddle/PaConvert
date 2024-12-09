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
        default=None,
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
        choices=["WARNING", "INFO", "DEBUG"],
        help="set log level, default is INFO",
    )
    parser.add_argument(
        "--log_markdown",
        default=False,
        type=bool,
        help="whether show log with markdown format",
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
    parser.add_argument(
        "--format",
        action="store_true",
        help="use black to format the converted code",
    )

    args = parser.parse_args()

    if args.run_check:
        cwd = os.path.dirname(__file__)
        converter = Converter()
        converter.run(cwd + "/example_code.py", cwd + "/temp_out/example_code.py")
        sys.exit(0)

    if args.separate_convert:
        project_num_100 = 0
        project_num_95 = 0
        project_num_90 = 0
        convert_rate_map = {}

        in_dir = os.path.abspath(args.in_dir)
        for project_name in os.listdir(in_dir):
            project_dir = os.path.join(in_dir, project_name)
            converter = Converter(show_unsupport=args.show_unsupport)
            converter.run(project_dir, args.out_dir, args.exclude_dirs)
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
        sys.exit(0)

    assert args.in_dir is not None, "User must specify --in_dir "
    converter = Converter(
        log_dir=args.log_dir,
        log_level=args.log_level,
        log_markdown=args.log_markdown,
        show_unsupport=args.show_unsupport,
        format=args.format,
    )
    converter.run(args.in_dir, args.out_dir, args.exclude_dirs)

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
