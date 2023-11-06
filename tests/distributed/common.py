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

import torch


def init_env():
    if "LOCAL_RANK" in os.environ:
        local_rank = os.environ["LOCAL_RANK"]
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
    torch.distributed.init_process_group(backend="nccl")


def dump_output(x):
    rank = torch.distributed.get_rank()
    path = os.environ.get("DUMP_FILE", None)
    if path is not None:
        path = path.replace("rank0", "rank" + str(rank))
        torch.save(x, path)


def run_local(cmd, envs=None):
    env_local = {}
    env_local.update(os.environ)
    if envs is not None:
        env_local.update(envs)
    print(f"local_cmd: {cmd}")

    local_proc = subprocess.Popen(
        cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env_local,
    )

    local_out, local_err = local_proc.communicate()
    sys.stdout.write("local_out: %s\n" % local_out.decode())
    sys.stderr.write("local_err: %s\n" % local_err.decode())
