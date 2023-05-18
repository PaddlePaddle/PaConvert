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

build_dir = "dist"
info_dir = "paconvert.egg-info"
expected_files = [
    "dist/paconvert-0.0.0.tar.gz",
    "dist/paconvert-0.0.0-py3-none-any.whl",
    "paconvert.egg-info/dependency_links.txt",
    "paconvert.egg-info/entry_points.txt",
    "paconvert.egg-info/PKG-INFO",
    "paconvert.egg-info/requires.txt",
    "paconvert.egg-info/SOURCES.txt",
    "paconvert.egg-info/top_level.txt",
]


def check_build_output():
    if not os.path.exists(build_dir):
        print(f"The dist directory '{build_dir}' is not existed!!")
        return False

    if not os.path.exists(info_dir):
        print(f"The info directory '{info_dir}' is not existed!!")
        return False

    for item in expected_files:
        if not os.path.exists(item):
            print(f"Missing file or directory: {item}")
            return False
        print(f"Got file or directory : {item}")
    return True


if not check_build_output:
    sys.exit(1)
