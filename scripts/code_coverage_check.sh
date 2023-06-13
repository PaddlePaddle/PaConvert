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

set +x

export FLAGS_set_to_1d=0
DEVELOP_IF="OFF"
ADD_GIT="OFF"
if [[ "$DEVELOP_IF" == "OFF" ]]; then
    cd /workspace/$2/PaConvert/
    PATH=$1
    echo "Insalling develop version paddle"
    python -m pip install --no-cache-dir  --force-reinstall paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
    python -c "import paddle; print('paddle version information:' ,paddle.__version__); print('paddle commit information:' ,paddle.__git_commit__)"
fi

# use Coverage diff-cover
echo "Insalling coverage and diff-cover for incremental code inspection"
pip install diff-cover coverage

if [[ "$DEVELOP_IF" == "ON" ]]; then
    pip install coverage diff-cover
    if [[ "$DEVELOP_IF" == "ON" ]]; then
        git remote add upstream https://github.com/PaddlePaddle/PaConvert
        git fetch upstream 
        git merge -X ours --allow-unrelated-histories upstream/master
    fi
fi

# coverage code check
echo '****************************start detecting coverate rate*********************************'
coverage run -m pytest

echo '**************************start generating coverage.xml file******************************'
coverage xml -o coverage.xml
coverage html

echo '************************start generating coverage rate report*****************************'
diff-cover coverage.xml --compare-branch origin/master > temp.txt;check_error1=$?

# Check the coverage results
cat temp.txt

echo '***********************start generating coverage rate detect******************************'
python  tools/coverage/coverage_diff.py;check_error2=$?

check_error=$((check_error1||check_error2)) 
echo "$check_error" > flag.txt

echo '*******************start generating coverage rate visualization************************\n'
