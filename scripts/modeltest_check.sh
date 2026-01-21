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

echo '************************************************************************************************************'
echo "Insalling latest release cpu version torch"
python -m pip install -U torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print('torch version information:' ,torch.__version__)"

echo '************************************************************************************************************'
echo "Insalling develop cpu version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
# For bypass broken update in paddle, should not merged into master
no_proxy="*" python -m pip install --force-reinstall --no-deps -U --pre paddlepaddle==3.4.0.dev20260119 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
# python -m pip install --force-reinstall --no-deps -U --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -c "import paddle; print('paddle version information:' , paddle.__version__); commit = paddle.__git_commit__;print('paddle commit information:' , commit)"

echo '************************************************************************************************************'
echo "Insalling paconvert requirements"
python -m pip install -r requirements.txt

echo '************************************************************************************************************'
echo 'Start modeltest'
python tools/modeltest/modeltest_check.py;check_error1=$?


echo '************************************************************************************************************'
echo "______      _____                          _   "
echo "| ___ \    / ____|                        | |  "
echo "| |_/ /_ _| |     ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  | |    / _ \\| \\_ \\ \\ / / _ \\ \\__| __|"
echo "| | | (_| | |___| (_) | | | \\ V /  __/ |  | |_ "
echo "\\_|  \\__,_|\\_____\\___/|_| |_|\\_/ \\___|_|   \\__|"
echo '************************************************************************************************************'
if [ ${check_error1} != 0  ]; then  
    echo "Your PR code model test check failed."
else
    echo "Your PR code model test check passed."
fi
echo '************************************************************************************************************'

exit ${check_error1}
