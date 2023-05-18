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

DEVELOP_IF="OFF"

if [[ "$DEVELOP_IF" == "OFF" ]]; then
    cd /workspace/$2/PaConvert/
    PATH=$1
fi

echo "start pipline testing..."
echo '*******************start generating source and wheel distribution*******************'

python tools/pipeline/build_and_deploy.py sdist bdist_wheel

echo '*******************start detecting source and wheel distribution*******************'

python tools/pipeline/file_detect.py;check_error=$?

rm -rf dist paconvert.egg-info build
echo '************************************************************************************'
echo "______                                   _   "
echo "| ___ \                                 | |  "
echo "| |_/ /_ _  ___ ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  |/ __/ _ \\| \_ \ \ / / _ \ \__| __|"
echo "| | | (_| | (_| (_) | | | \\ V /  __/ |  | |_ "  
echo "\\_|  \\__,_|\\___\\___/|_| |_|\\_/ \\___|_|   \\__|"  
echo -e '\n************************************************************************************' 
if [ ${check_error} != 0 ];then
    echo "Your PR code pipeline test check failed." 
else
    echo "Your PR code pipeline test check passed."
fi
echo '************************************************************************************'

exit ${check_error}

                                             
                                             
