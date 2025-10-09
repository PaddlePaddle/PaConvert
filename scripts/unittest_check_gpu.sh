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

echo '************************************************************************************************************'
echo "Insalling latest release gpu version torch"
python -m pip uninstall -y torchaudio
python -m pip install torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl
python -m pip install torchvision-0.22.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl
# python -m pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('torch version information:' ,torch.__version__)"

echo "Installing transformers from git develop branch"
python -m pip install -U git+https://github.com/huggingface/transformers.git
python -c "import transformers; print('transformers version information:', transformers.__version__)"

echo "Installing paddleformers from git develop branch"
python -m pip install -U git+https://github.com/PaddlePaddle/PaddleFormers.git
python -c "import paddleformers; print('paddleformers version information:', paddleformers.__version__)"

echo '************************************************************************************************************'
echo "Insalling develop gpu version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
python -m pip install --force-reinstall --no-deps -U --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
python -c "import paddle; print('paddle version information:' , paddle.__version__); commit = paddle.__git_commit__;print('paddle commit information:' , commit)"

echo '************************************************************************************************************'
echo "Insalling paconvert requirements"
python -m pip install -r requirements.txt

echo '************************************************************************************************************'
echo "Checking code unit test by pytest ..."
python -m pip install pytest-timeout pytest-xdist pytest-rerunfailures
python -m pytest -n 1 --reruns=3 ./tests; check_error=$?
if [ ${check_error} != 0 ];then
    echo "Rerun unit test check." 
    python -m pytest -n 1 --lf ./tests; check_error=$?
fi

echo '************************************************************************************************************'
echo "______      _____                          _   "
echo "| ___ \    / ____|                        | |  "
echo "| |_/ /_ _| |     ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  | |    / _ \\| \\_ \\ \\ / / _ \\ \\__| __|"
echo "| | | (_| | |___| (_) | | | \\ V /  __/ |  | |_ "
echo "\\_|  \\__,_|\\_____\\___/|_| |_|\\_/ \\___|_|   \\__|"
echo '************************************************************************************************************'
if [ ${check_error} != 0 ];then
    echo "Your PR code unit test check failed." 
    echo "Please run the following command." 
    echo "" 
    echo "    python -m pytest tests" 
    echo "" 
    echo "For more information, please refer to our check guide:" 
    echo "https://github.com/PaddlePaddle/PaConvert#readme." 
else
    echo "Your PR code unit test check passed."
fi
echo '************************************************************************************************************'

exit ${check_error}
