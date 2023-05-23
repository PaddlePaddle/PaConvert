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

DEVELOP_IF="OFF"

if [[ "$DEVELOP_IF" == "OFF" ]]; then
    cd /workspace/$2/PaConvert/
    PATH=$1
else
    # use pre-commit 2.17
    if ! [[ $(pre-commit --version) == *"2.17.0"* ]]; then
        pip install pre-commit==2.17.0 1>nul
    fi
fi

# pre-commit multi-thread running.
echo "Checking code style by pre-commit ..."
pre-commit run --all-files;check_error=$?
echo '************************************************************************************'
echo "______      _____                          _   "
echo "| ___ \    / ____|                        | |  "
echo "| |_/ /_ _| |     ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  | |    / _ \\| \\_ \\ \\ / / _ \\ \\__| __|"
echo "| | | (_| | |___| (_) | | | \\ V /  __/ |  | |_ "
echo "\\_|  \\__,_|\\_____\\___/|_| |_|\\_/ \\___|_|   \\__|"
echo -e '\n************************************************************************************'
if [ ${check_error} != 0 ];then
    echo "Your PR code style check failed."
    echo "Please install pre-commit locally and set up git hook scripts:"
    echo ""
    echo "    pip install pre-commit==2.17.0"
    echo "    pre-commit install"
    echo ""
    echo "Then, run pre-commit to check codestyle issues in your PR:"
    echo ""
    echo "    pre-commit run --all-files"
    echo ""
    echo "For more information, please refer to our codestyle check guide:"
    echo "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/codestyle_check_guide_cn.html"
else
    echo "Your PR code style check passed."
fi
echo '************************************************************************************'

exit ${check_error}
