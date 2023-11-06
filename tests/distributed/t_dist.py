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

import os
import sys
import tempfile

import paddle
import torch
from common import run_local

sys.path.append(os.path.dirname(__file__) + "/..")
from apibase import APIBase


def get_dump_file(type, temp_dir):
    return f"{temp_dir}/out_{type}_{os.getpid()}_rank0.dat"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("args: [pytorch file] [paddle file]")
        sys.exit()

    temp_dir = tempfile.TemporaryDirectory()
    pytorch_ret = []
    paddle_ret = []

    dump_file = get_dump_file("pytorch", temp_dir.name)
    run_local(sys.argv[1], {"DUMP_FILE": dump_file})
    try:
        pytorch_ret = torch.load(dump_file)
    except Exception as e:
        print(e)

    dump_file = get_dump_file("paddle", temp_dir.name)
    run_local(sys.argv[2], {"DUMP_FILE": dump_file})
    try:
        paddle_ret = paddle.load(dump_file)
    except Exception as e:
        print(e)

    apibase = APIBase("torch.nn.distributed")
    apibase.compare("torch.nn.distributed", pytorch_ret, paddle_ret)
    print(pytorch_ret, paddle_ret)

    temp_dir.cleanup()
