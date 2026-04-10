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

# GPU 脚本之前没有像 CPU 脚本一样固定 cwd，导致它更依赖外部调用方从哪里触发。
# 这里统一切到 repo root，是为了让 requirements.txt、./tests 以及 apibase 的临时文件路径
# 都基于同一个仓库目录；如果 workspace 或 $1 不对，就立刻失败并给出明确日志。
cd /workspace/$1/PaConvert/ || {
    echo "[unittest-gpu] Failed to enter repo root: /workspace/$1/PaConvert/"
    exit 1
}

# 这行日志用于在 CI 首屏确认 GPU job 是否真的进入了预期目录，
# 排查“同一脚本在不同入口目录下行为不一致”的问题时会更直观。
echo "Current working directory: $(pwd)"

# 与 CPU 脚本相同，先做最小环境自检：
# - requirements.txt 决定依赖安装是否可执行
# - tests 目录决定 pytest 收集是否能落在正确位置
# 如果这两个路径不可见，就说明当前工作区本身已经不满足执行前提。
test -f requirements.txt || {
    echo "[unittest-gpu] requirements.txt not found under repo root"
    exit 1
}

test -d tests || {
    echo "[unittest-gpu] tests directory not found under repo root"
    exit 1
}

# These files mutate process-level defaults and are more stable when run in
# their own pytest process after the main suite.
# 这些文件会修改进程级默认状态，是引发全量执行顺序污染的高风险集合。
# 处理方式与 CPU 一致：
# 1. 主套件先忽略它们
# 2. 再逐个单文件、单独进程执行
# 这样既不丢掉覆盖率，也能把“普通测试”和“全局状态测试”的影响面拆开观察。
ISOLATED_TEST_FILES=(
  ./tests/test_set_default_device.py
  ./tests/test_set_default_dtype.py
  ./tests/test_set_default_tensor_type.py
  ./tests/test_set_num_threads.py
  ./tests/test_set_printoptions.py
)

# 统一生成 pytest --ignore 参数，避免人工维护两套重复命令。
IGNORE_ARGS=()
for test_file in "${ISOLATED_TEST_FILES[@]}"; do
    IGNORE_ARGS+=("--ignore=${test_file}")
done

echo '************************************************************************************************************'
echo "Insalling latest release gpu version torch"
python -m pip uninstall -y torchaudio
python -m pip install torch-2.7.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl
python -m pip install torchvision-0.22.1+cu118-cp310-cp310-manylinux_2_28_x86_64.whl
# python -m pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('torch version information:' ,torch.__version__)"

echo '************************************************************************************************************'
echo "Insalling develop gpu version paddle"
python -m pip uninstall -y paddlepaddle
python -m pip uninstall -y paddlepaddle-gpu
python -m pip install --force-reinstall --no-deps -U --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
python -m pip install safetensors==0.6.2
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
echo "Checking code gpu unit test by pytest ..."
# GPU 这边保留 pytest-xdist，因为脚本原本就在使用 -n 1 的执行方式；
# 但去掉 pytest-rerunfailures/--reruns=3，避免 first-run 失败被自动重试掩盖。
python -m pip install pytest-timeout pytest-xdist
# 这里同样显式设置 PYTHONPATH=.:tests，确保 GPU CI 在脚本直接调用 python -m pytest 时，
# 对 tests/apibase.py 的导入与 pytest.ini 中的配置保持一致。
PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings -n 1 "${IGNORE_ARGS[@]}" ./tests
first_run_error=$?

# 首轮结果决定最终成败；失败后保留一次 --lf rerun 仅用于诊断，
# 方便观察失败是否与执行顺序或环境状态相关，但不会再改变最终 exit code。
if [ ${first_run_error} != 0 ]; then
    echo "[unittest-gpu] Diagnostic rerun of failed tests. This does not change the final result."
    PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings -n 1 "${IGNORE_ARGS[@]}" --lf ./tests || true
fi

# 逐个执行隔离文件，让每个 stateful 文件都跑在新的 pytest 进程里。
# 这样即使某个文件会修改默认 device / dtype，也不会把副作用泄漏到其他测试文件。
isolated_error=0
for test_file in "${ISOLATED_TEST_FILES[@]}"; do
    echo "[unittest-gpu] Running isolated test file: ${test_file}"
    PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings -n 1 "${test_file}"
    file_error=$?

    if [ ${file_error} != 0 ]; then
        echo "[unittest-gpu] Diagnostic rerun for isolated file: ${test_file}. This does not change the final result."
        PYTHONPATH=.:tests python -m pytest -v -s -p no:warnings -n 1 "${test_file}" || true
        isolated_error=1
    fi
done

# 最终退出码仍然只看“主套件 + 隔离套件”的 first-run 结果。
# 任何一边失败都应该让 job fail，这样 CI 才能真实暴露不稳定问题。
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
    echo "Your PR code gpu unit test check failed." 
    echo "Please run the following command." 
    echo "" 
    echo "    python -m pytest tests" 
    echo "" 
    echo "For more information, please refer to our check guide:" 
    echo "https://github.com/PaddlePaddle/PaConvert#readme." 
else
    echo "Your PR code gpu unit test check passed."
fi
echo '************************************************************************************************************'

exit ${check_error}
