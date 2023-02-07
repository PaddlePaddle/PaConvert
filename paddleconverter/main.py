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

from paddleconverter.converter import Converter

def main():
    parser = argparse.ArgumentParser(prog="paddleconverter", description="Paddleconverter tool entry point")
    parser.add_argument("--in_dir", default=None, type=str, help='the input PyTorch file or directory.')
    parser.add_argument("--out_dir", default=None, type=str, help='the output Paddle directory.')
    parser.add_argument("--exclude_dir", default=None, type=str, help='the exclude Pytorch file or directory, which will not be converted.')
    parser.add_argument("--log_dir", default=None, type=str, help='the input PyTorch file or directory.')
    parser.add_argument("--log_level", default="INFO", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="set log level, default is INFO")
    parser.add_argument("--run_check", default=None, type=str, help='run check the paddle convert tool')

    args = parser.parse_args()

    if args.run_check is not None:
        cwd = os.path.dirname(__file__)
        coverter = Converter(args.log_dir, args.log_level)
        
        success_api, failed_api = coverter.run(cwd + '/tests/test_model.py', cwd +'/tests/temp_out/test_model.py')
        
        if success_api==90 and failed_api == 3:
            logging.info("Run check successfully! Use 'paddleconverter --in_dir IN_DIR --out_dir OUT_DIR' to convert your project ")
            sys.exit(0)
        else:
            logging.warning("Run check wrong! Use 'paddleconverter --in_dir IN_DIR --out_dir OUT_DIR' to convert your project ")
            sys.exit(1)
        

    assert args.in_dir is not None, "User must specify --in_dir "
    assert args.out_dir is not None, "User must specify --out_dir "
    assert args.out_dir != args.in_dir, "--out_dir must be different from --in_dir"
    
    coverter = Converter(args.log_dir, args.log_level)
    coverter.run(args.in_dir, args.out_dir, args.exclude_dir)


if __name__ == "__main__":
    main()
