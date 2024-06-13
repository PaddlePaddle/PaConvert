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

CODE_CONSISTENCY_MAPPING = {}
cur_dir = os.path.dirname(__file__)

def add_to_dict(torch_file, paddle_file):
    if not os.path.isabs(torch_file):
        torch_file = os.path.join(cur_dir, "torch_code/", torch_file)
        
    if not os.path.isabs(paddle_file):
        paddle_file = os.path.join(cur_dir, "paddle_code/", paddle_file)

    global CODE_CONSISTENCY_MAPPING
    CODE_CONSISTENCY_MAPPING[torch_file] = paddle_file


# this part is about api mapping file
add_to_dict("api_torch_equal.py", "api_paddle_equall_all.py")
add_to_dict("api_torch_randint.py", "api_paddle_randint.py")
add_to_dict("api_torch_LongTensor.py", "api_paddle_Tensor2Long.py")
add_to_dict("api_torch_sigmoid.py", "api_paddle_nn_functional_sigmoid.py")
add_to_dict("api_torch_Tensor_to.py", "api_paddle_Tensor_cast.py")
add_to_dict("api_torch_Tensor_new_full.py", "api_paddle_Tensor_new_full.py")
add_to_dict("api_torch_Generator.py", "api_paddle_Generator.py")
add_to_dict("api_torch_random_manual_seed.py", "api_paddle_seed.py")
add_to_dict("api_torch_nn_BatchNorm3d.py", "api_paddle_nn_BatchNorm3D.py")
add_to_dict("api_torch_Tensor_new_ones.py", "api_paddle_Tensor_new_ones.py")
add_to_dict("api_torch_Tensor_new_tensor.py", "api_paddle_new_tensor.py")
add_to_dict("api_torch_version.py", "api_paddle_version.py")
add_to_dict("api_torch_Tensor_expand.py", "api_paddle_Tensor_expand.py")
add_to_dict("api_torch_FloatTensor.py", "api_paddle_Tensor2Float.py")
add_to_dict("api_torch_Tensor_normal_.py", "api_paddle_Tensor_normal_.py")
add_to_dict("api_torch_nn_BatchNorm1d.py", "api_paddle_nn_BatchNorm1D.py")
add_to_dict("api_torch_Tensor_uniform_.py", "api_paddle_Tensor_uniform_.py")
add_to_dict("api_torch_sigmoid.py", "api_paddle_nn_functional_sigmoid.py")
add_to_dict("api_torch_nn_BCEWithLogitsLoss.py", "api_paddle_nn_BCEWithLogitsLoss.py")
add_to_dict(
    "api_torch_nn_functional_interpolate.py", "api_paddle_nn_functional_interpolate.py"
)
add_to_dict("api_torch_new_empty.py", "api_paddle_new_empty.py")
add_to_dict("api_torch_permute.py", "api_paddle_transpose.py")
add_to_dict("api_torch_tensor.py", "api_paddle_to_tensor.py")
add_to_dict("api_torch_Size.py", "api_paddle_Size.py")
add_to_dict("api_torch_nn_InstanceNorm3d.py", "api_paddle_nn_InstanceNorm3D.py")
add_to_dict("api_torch_index_copy_.py", "api_paddle_index_copy_.py")
add_to_dict("api_torch_Tensor_new_zeros.py", "api_paddle_new_zeros.py.py")
add_to_dict("api_torch_cuda_is_available.py", "api_paddle_cuda_is_available.py")
add_to_dict("api_torch_nn_BatchNorm2d.py", "api_paddle_nn_BatchNorm2D.py")
add_to_dict("api_torch_IntTensor.py", "api_paddle_Tensor2Int.py")
add_to_dict("api_torch_data_BatchSampler.py", "api_paddle_io_BatchSampler.py")
add_to_dict("non_torch_class_method.py", "non_paddle_class_method.py")
add_to_dict("class_method_static_call.py", "class_method_static_call.py")
add_to_dict("import_analysis.py", "import_analysis.py")
add_to_dict("torch_llama.py", "paddle_llama.py")
add_to_dict("may_torch_package_list.py", "may_paddle_package_list.py")
add_to_dict("mark_unsupport.py", "mark_unsupport.py")
add_to_dict("transformers_Qwen.py", "paddlenlp_Qwen.py")
add_to_dict("type_hinting.py", "type_hinting.py")
add_to_dict("alias_call.py", "alias_call.py")


# this part is about attribute mapping file

add_to_dict(
    "attribute_torch_Tensor_requires_grad.py",
    "attribute_paddle_Tensor_stop_gradient.py",
)

add_to_dict(
    "attribute_visit_name.py",
    "attribute_visit_name.py",
)

add_to_dict(
    "attribute_setattr.py",
    "attribute_setattr.py",
)


# this part is about custom op
cur_dir = os.path.dirname(__file__)
add_to_dict(os.path.join(cur_dir, "custom_op/torch_code/"), os.path.join(cur_dir, "custom_op/paddle_code/"))
