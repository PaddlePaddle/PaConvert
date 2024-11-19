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

cd /workspace/$1/PaConvert/

# use Coverage diff-cover
echo "Insalling coverage and diff-cover for incremental code inspection"
python -m pip install coverage diff-cover
python -m pip install pytest-timeout

if [[ "$DEVELOP_IF" == "ON" ]]; then
    python -m pip install coverage diff-cover
    if [[ "$DEVELOP_IF" == "ON" ]]; then
        git remote add upstream https://github.com/PaddlePaddle/PaConvert
        git fetch upstream 
        git merge -X ours --allow-unrelated-histories upstream/master
    fi
fi

# coverage code check
echo '****************************start detecting coverate rate*********************************'
python -m coverage run -m pytest

echo '**************************start generating coverage.xml file******************************'
python -m coverage xml -o coverage.xml
python -m coverage html

echo '************************start generating coverage rate report*****************************'
diff-cover coverage.xml --compare-branch origin/master > temp.txt;check_error1=$?

# Check the coverage results
cat temp.txt

echo '***********************start generating coverage rate detect******************************'
python tools/coverage/coverage_diff.py;check_error2=$?

check_error=$((check_error1||check_error2)) 
echo "$check_error" > flag.txt

echo '*******************start generating coverage rate visualization************************\n'
