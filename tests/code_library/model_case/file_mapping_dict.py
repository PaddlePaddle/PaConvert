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

global_file_mapping_dict = {}
dir_name = os.path.dirname(__file__)
torch_file_path = os.path.join(dir_name, "torch_code/")
paddle_file_path = os.path.join(dir_name, "paddle_code/")


def add_to_dict(torch_file, paddle_file):
    global global_file_mapping_dict
    torch_file = os.path.join(torch_file_path, torch_file)
    paddle_file = os.path.join(paddle_file_path, paddle_file)
    global_file_mapping_dict[torch_file] = paddle_file


# this part is about model mapping file
# add_to_dict("model_torch_mobilenet.py", "model_paddle_mobilenet.py")
add_to_dict("model_torch_resnet.py", "model_paddle_resnet.py")
# add_to_dict("model_torch_vggnet.py", "model_paddle_vggnet.py")
# add_to_dict("model_torch_xception.py", "model_paddle_xception.py")
add_to_dict("model_torch_lenet.py", "model_paddle_lenet.py")
