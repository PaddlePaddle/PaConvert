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

# 先显式进入 repo root，而不是继续依赖外部调用方的当前目录。
# 这样后续所有相对路径都会稳定落在同一个工作区下，包括：
# 1. requirements.txt 的安装路径
# 2. pytest ./tests 的收集根目录
# 3. tests/apibase.py 中基于 os.getcwd() 生成的临时文件目录
# 如果这里进入失败，就直接中断，避免后面出现更难读的连锁报错。
cd /workspace/$1/PaConvert/ || {
    echo "[unittest-cpu] Failed to enter repo root: /workspace/$1/PaConvert/"
    exit 1
}

# 这两个检查属于 fail-fast：
# 1. requirements.txt 用于后续安装测试依赖
# 2. tests 是 unittest 主入口
# 如果这两个路径在当前目录下都不可见，说明 workspace 本身或 cwd 已经不对，
# 继续跑只会把真正的问题伪装成 pip/pytest 的噪声错误。
test -f requirements.txt || {
    echo "[unittest-cpu] requirements.txt not found under repo root"
    exit 1
}

test -d tests || {
    echo "[unittest-cpu] tests directory not found under repo root"
    exit 1
}

# These files mutate process-level defaults and are more stable when run in
# their own pytest process after the main suite.
# 这些文件会修改默认 device / dtype / tensor type / 线程数 / printoptions 等进程级状态。
# 本次改法分两步：
# 1. 主套件先通过 --ignore 跳过它们，只验证普通测试
# 2. 再为每个文件单独启动一个 pytest 进程执行
# 目的不是跳过这些测试，而是通过进程边界隔离副作用，降低 first-run 全量执行时
# 因状态污染导致的“第一次失败、rerun 又通过”的概率。
ISOLATED_TEST_FILES=(
  ./tests/test_set_default_device.py
  ./tests/test_set_default_dtype.py
  ./tests/test_set_default_tensor_type.py
  ./tests/test_set_num_threads.py
  ./tests/test_set_printoptions.py
)

# 把上面的列表转换成 pytest 所需的 --ignore 参数。
# 这样做可以避免手写重复命令，同时确保 CPU/GPU 两个脚本使用完全一致的隔离集合。
IGNORE_ARGS=()
for test_file in "${ISOLATED_TEST_FILES[@]}"; do
    IGNORE_ARGS+=("--ignore=${test_file}")
done

echo '************************************************************************************************************'
echo "Insalling latest release cpu version torch"
python -m pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print('torch version information:' ,torch.__version__)"

echo '************************************************************************************************************'
echo "Insalling develop cpu version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
python -m pip install paddlepaddle-0.0.0-cp39-cp39-linux_x86_64.whl
# python -m pip install --force-reinstall --no-deps -U --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -c "import paddle; print('paddle version information:' , paddle.__version__); commit = paddle.__git_commit__;print('paddle commit information:' , commit)"

echo '************************************************************************************************************'
#echo "Installing transformers==4.55.4"
#python -m pip install transformers==4.55.4
#python -c "import transformers; print('transformers version information:', transformers.__version__)"

echo '************************************************************************************************************'
#echo "Installing paddleformers from git develop branch"
#python -m pip install -U git+https://github.com/PaddlePaddle/PaddleFormers.git
#python -c "import paddleformers; print('paddleformers version information:', paddleformers.__version__)"


echo '************************************************************************************************************'
echo "Insalling paconvert requirements"
python -m pip install -r requirements.txt

echo '************************************************************************************************************'
echo "Checking code cpu unit test by pytest ..."
python -m pip install pytest-timeout
# 这里显式设置 PYTHONPATH=.:tests，有两个目的：
# 1. 让脚本直接调用 python -m pytest 时，也能稳定导入 tests/apibase.py
# 2. 与 pytest.ini 中 pythonpath = tests 的配置形成显式兜底
# 主套件先忽略掉 stateful 文件，把“普通测试是否稳定”单独跑出来看。
PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings "${IGNORE_ARGS[@]}" ./tests
first_run_error=$?

# 首轮结果决定最终 success / fail。
# 如果首轮失败，保留一次 --lf rerun 仅用于诊断日志，帮助判断失败是否具有顺序相关性；
# 但 rerun 结果不会再覆盖首轮退出码，避免把真实问题“洗绿”。
if [ ${first_run_error} != 0 ]; then
    echo "[unittest-cpu] Diagnostic rerun of failed tests. This does not change the final result."
    PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings "${IGNORE_ARGS[@]}" --lf ./tests || true
fi

# 接下来逐个执行隔离文件。每个文件都会启动新的 pytest 进程：
# 1. 前一个文件留下的全局状态不会泄漏到下一个文件
# 2. 一旦某个隔离文件失败，再追加一次诊断性 rerun 方便看日志
# 3. isolated_error 只记录“是否存在任一隔离文件失败”，最后与主套件统一汇总
isolated_error=0
for test_file in "${ISOLATED_TEST_FILES[@]}"; do
    echo "[unittest-cpu] Running isolated test file: ${test_file}"
    PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings "${test_file}"
    file_error=$?

    if [ ${file_error} != 0 ]; then
        echo "[unittest-cpu] Diagnostic rerun for isolated file: ${test_file}. This does not change the final result."
        PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings "${test_file}" || true
        isolated_error=1
    fi
done

# 最终退出码采用主套件和隔离套件的并集语义：
# - 任意一边失败，整个 unittest job 都失败
# - 不因为诊断 rerun 通过而掩盖 first-run 的真实结果
check_error=0
if [ ${first_run_error} != 0 ] || [ ${isolated_error} != 0 ]; then
    check_error=1
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
    echo "Your PR code cpu unit test check failed." 
    echo "Please run the following command." 
    echo "" 
    echo "    python -m pytest tests" 
    echo "" 
    echo "For more information, please refer to our check guide:" 
    echo "https://github.com/PaddlePaddle/PaConvert#readme." 
else
    echo "Your PR code cpu unit test check passed."
fi
echo '************************************************************************************************************'

exit ${check_error}
