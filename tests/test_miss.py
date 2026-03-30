# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from apibase import APIBase


def test(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = []
    for item in data:
        api = item["api"]
        obj = APIBase(api)
        torch_codes = item["test_code"]
        excepted_outputs = item["excepted_paddle_code"]
        for i in range(len(torch_codes)):
            code = torch_codes[i]
            excepted_output = excepted_outputs[i]
            obj.run(pytorch_code=code, expect_paddle_code=excepted_output, mode="min")


path = "tests/test_miss.json"
test(path)
