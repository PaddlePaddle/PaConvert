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


echo "Insalling develop version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
rm -rf /root/anaconda3/lib/python*/site-packages/paddlepaddle-0.0.0.dist-info/
python -m pip install --no-cache-dir paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
python -c "import paddle; print('paddle version information:' , paddle.__version__); commit = paddle.__git_commit__;print('paddle commit information:' , commit)"

python ../../paconvert/main.py --in_dir . --out_dir /tmp/paddle --log_level "DEBUG"

export CUDA_VISIBLE_DEVICES=0,1

if [ $# -gt 0 ] ; then
    item=$1
    cmd1="torchrun --nproc_per_node=2 ${item}"
    cmd2="python -m paddle.distributed.launch /tmp/paddle/${item}"
    python t_dist.py "$cmd1" "$cmd2"
    exit
fi

check_error=0

failed_tests=()

test_list=`ls *.py | grep -v common.py | grep -v t_dist.py`
for item in $test_list; do
    cmd1="torchrun --nproc_per_node=2 ${item}"
    cmd2="python -m paddle.distributed.launch /tmp/paddle/${item}"
    python t_dist.py "$cmd1" "$cmd2"; tmp_check_error=$?
    if [ $tmp_check_error -ne 0 ]; then
        check_error=1
        failed_tests+=("$item")
    fi
done

echo "Failed tests:"
for item in $my_list; do
    echo "$item"
done
echo "Exit code:" $check_error
exit $check_error
