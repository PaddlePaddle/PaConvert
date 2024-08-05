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

export LD_LIBRARY_PATH=/root/anaconda3/lib:$LD_LIBRARY_PATH

echo "Insalling cpu version torch"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print('torch version information:' ,torch.__version__)"

echo "Insalling develop version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
rm -rf /root/anaconda3/lib/python*/site-packages/paddlepaddle-0.0.0.dist-info/
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
python -c "import paddle; print('paddle version information:' , paddle.__version__); commit = paddle.__git_commit__;print('paddle commit information:' , commit)"

echo "Checking code unit test by pytest ..."
python -m pip install pytest-timeout pytest-xdist pytest-rerunfailures
python -m pytest -n 1 --reruns=3 ./tests; check_error=$?
if [ ${check_error} != 0 ];then
    echo "Rerun unit test check." 
    python -m pytest --lf  -n 1 ./tests; check_error=$?
fi

echo '************************************************************************************'
echo "______      _____                          _   "
echo "| ___ \    / ____|                        | |  "
echo "| |_/ /_ _| |     ___  _ ____   _____ _ __| |_ "
echo "|  __/ _  | |    / _ \\| \\_ \\ \\ / / _ \\ \\__| __|"
echo "| | | (_| | |___| (_) | | | \\ V /  __/ |  | |_ "
echo "\\_|  \\__,_|\\_____\\___/|_| |_|\\_/ \\___|_|   \\__|"
echo -e '\n************************************************************************************' 
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
echo '************************************************************************************'

exit ${check_error}

                                             
                                             
