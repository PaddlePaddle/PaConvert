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

python ../../paconvert/main.py --in_dir . --out_dir /tmp/paddle --log_level "DEBUG"

export CUDA_VISIBLE_DEVICES=0,1

if [ $# -gt 0 ] ; then
    item=$1
    torchrun --nproc_per_node=2 ${item}
    python -m paddle.distributed.launch /tmp/paddle/${item}
    exit
fi

test_list="scatter reduce_scatter scatter_object_list all_to_all"
for i in $test_list; do
    test_file="${i}.py"
    torchrun --nproc_per_node=2 ${test_file}
    python -m paddle.distributed.launch /tmp/paddle/${test_file}
done
