# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import random
import subprocess

import pytest

failed_tests = []
# add a marker to each test that should run on GPU 0 or 1


def pytest_collection_modifyitems(session, config, items):
    num_gpus = 2
    tests_per_gpu = len(items) // num_gpus
    random.shuffle(items)
    # 分配测试到两个列表
    gpu0_tests = items[:tests_per_gpu]
    gpu1_tests = items[tests_per_gpu:]

    # 为每个测试添加一个标记，表明它应该在哪个GPU上运行
    for i, item in enumerate(gpu0_tests):
        item.add_marker(pytest.mark.gpu0)
    for i, item in enumerate(gpu1_tests):
        item.add_marker(pytest.mark.gpu1)


def pytest_runtest_logreport(report):
    if report.failed:
        failed_tests.append(report.nodeid)


def rerun_failed_tests_with_gpu():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    cmd = ["pytest", "-n", "1", "-k", "or".join(failed_tests)]
    subprocess.run(cmd, env=env, shell=True)


def pytest_sessionfinish(session, exitstatus):
    if exitstatus != 0:  # 如果有失败的测试,独占两个GPU,运行失败的测试
        rerun_failed_tests_with_gpu()
