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
import subprocess
import sys
import tempfile

import paddle
import torch

sys.path.append(os.path.dirname(__file__) + "/..")
from apibase import APIBase

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("must input args: [pytorch file] [paddle file]")
        exit()

    pytorch_cmd = sys.argv[1]
    paddle_cmd = sys.argv[2]
    print(f"pytorch cmd: {pytorch_cmd}")
    print(f"paddle cmd: {paddle_cmd}")
    temp_dir = tempfile.TemporaryDirectory()

    torch_ret_file = f"{temp_dir.name}/out_torch_{os.getpid()}.dat"
    local_proc = subprocess.Popen(
        pytorch_cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.update({"DUMP_FILE": torch_ret_file}),
    )
    local_out, local_err = local_proc.communicate()
    sys.stdout.write("torch local_out: %s\n" % local_out.decode())
    sys.stderr.write("torch local_err: %s\n" % local_err.decode())
    exit_code = local_proc.returncode
    if exit_code != 0:
        exit(exit_code)

    paddle_ret_file = f"{temp_dir.name}/out_paddle_{os.getpid()}.dat"
    local_proc = subprocess.Popen(
        paddle_cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.update({"DUMP_FILE": paddle_ret_file}),
    )
    local_out, local_err = local_proc.communicate()
    sys.stdout.write("paddle local_out: %s\n" % local_out.decode())
    sys.stderr.write("paddle local_err: %s\n" % local_err.decode())
    exit_code = local_proc.returncode
    if exit_code != 0:
        exit(exit_code)

    if os.path.exists(torch_ret_file) and os.path.exists(paddle_ret_file):
        pytorch_ret = torch.load(torch_ret_file)
        paddle_ret = paddle.load(paddle_ret_file)
        apibase = APIBase("torch.nn.distributed")
        apibase.compare("torch.nn.distributed", pytorch_ret, paddle_ret)

    temp_dir.cleanup()
