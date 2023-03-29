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

import os
import logging
import argparse
import sys
sys.path.append(os.path.dirname(__file__) + '/..')

from paconvert.converter import Converter

def main():
    if sys.version_info < (3, 8):
        raise RuntimeError(
            "PaConvert use new AST syntax and only supports Python version >= 3.8 now.")
            
    parser = argparse.ArgumentParser(prog="paconvert", description="PaConverter tool entry point")
    parser.add_argument("--in_dir", default='./tests/test_transpose.py', type=str, help='the input PyTorch file or directory.')
    parser.add_argument("--out_dir", default=None, type=str, help='the output Paddle directory.')
    parser.add_argument("--exclude_dirs", default=None, type=str, help='the exclude Pytorch file or directory, which will not be converted.')
    parser.add_argument("--log_dir", default=None, type=str, help='the input PyTorch file or directory.')
    parser.add_argument("--log_level", default="INFO", type=str, choices=["DEBUG", "INFO"], help="set log level, default is INFO")
    parser.add_argument("--run_check", default=False, type=bool, help='run check the paddle convert tool')
    parser.add_argument("--show_unsupport", default=False, type=bool, help='show these APIs which are not supported to convert now')

    args = parser.parse_args()

    if args.run_check:
        cwd = os.path.dirname(__file__)
        coverter = Converter(args.log_dir, args.log_level, args.show_unsupport)
        coverter.run(cwd + '/example_code.py', cwd +'/temp_out/example_code.py')
        sys.exit(0)

    assert args.in_dir is not None, "User must specify --in_dir "
    coverter = Converter(args.log_dir, args.log_level, args.show_unsupport)
    coverter.run(args.in_dir, args.out_dir, args.exclude_dirs)


if __name__ == "__main__":
    main()
