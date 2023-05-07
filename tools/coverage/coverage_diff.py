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

number = 90


def _check_coverage_rate():
    """ "Run it every time and if coverage rate is too low,
    it warns the user and updates the coverage data"""

    result = []
    flag = 100
    with open("./temp.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("Coverage:"):
                flag = int(line[-4:-1])
            if line.endswith("%)"):
                temp_percent = int(line.split(" ")[1][1:-2])
                if temp_percent < number:
                    result.append(line)

    os.system("rm temp.txt")

    if flag > number:
        return False

    return True


if __name__ == "__main__":
    flag = _check_coverage_rate()

    if flag:
        sys.exit(1)
