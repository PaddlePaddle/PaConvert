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

MODEL_LIST = []

def add_model(torch_file):
    dir_name = os.path.dirname(__file__)
    torch_file = os.path.join(dir_name, "torch_code/", torch_file)
    global MODEL_LIST
    MODEL_LIST.append(torch_file)


# this part is about model file list
add_model("exclude_convert.py")
# add_model("model_torch_mobilenet.py")
add_model("model_torch_resnet.py")
# add_model("model_torch_vggnet.py")
# add_model("model_torch_xception.py")
add_model("model_torch_lenet.py")
