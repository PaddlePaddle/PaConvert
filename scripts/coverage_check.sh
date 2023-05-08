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

# use Coverage diff-cover
echo "Insalling coverage and diff-cover for incremental code inspection"
pip install coverage diff-cover


# coverage code check
coverage run -m pytest
coverage xml -o coverage.xml

git --version
git config --global user.name "paddle-ci"
git config --global user.email paddle-ci@baidu.com
git remote add upstream https://github.com/PaddlePaddle/PaConvert
git fetch upstream 
git merge -X ours --allow-unrelated-histories upstream/master;check_error1=$?
diff-cover coverage.xml --compare-branch upstream/master > temp.txt;check_error2=$?

# Check the coverage results
cat temp.txt

python tools/coverage/coverage_diff.py;check_error3=$?


echo -e '\n************************************************************************************'
if [ ${check_error1} != 0 ] || [ ${check_error2} != 0 ] || [ ${check_error3} != 0 ];then
    echo "Your PR code coverage rate check failed."
else
    echo "Your PR code coverage rate check passed."
fi
echo -e '************************************************************************************\n'

check_error=$((check_error1 || check_error2 || check_error3))
exit ${check_error}
