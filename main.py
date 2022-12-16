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
import argparse
import logging

from converter import Converter

def main():
    parser = argparse.ArgumentParser(prog="paddleconverter", description="Paddleconverter tool entry point")
    parser.add_argument("--in_dir", required=True, type=str, help='the input PyTorch file or directory.')
    parser.add_argument("--out_dir", required=True, type=str, help='the output Paddle directory.')
    parser.add_argument("--log_dir", default=None, type=str, help='the input PyTorch file or directory.')

    args = parser.parse_args()

    assert args.out_dir != args.in_dir, "--out_dir must be different from --in_dir"

    coverter = Converter(args.log_dir)
    coverter.run(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
