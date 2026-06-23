# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
# 

set -eo pipefail

DIST_OUT="$(pwd)/paddle_dist"

echo '******************************************************************************'
echo "Installing develop GPU version paddle"
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true
python -m pip install --force-reinstall --no-cache-dir -U --pre paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 120 --retries 3
python -m pip install safetensors==0.6.2
python -c "import paddle; print('paddle version: ', paddle.__version__); print('paddle commit info: ', paddle.__git_commit__)"

echo '******************************************************************************'
echo "Installing paconvert requirements"

cd tests/distributed

echo '******************************************************************************'
echo 'Converting torch code to paddle -> ${DIST_OUT}'
rm -rf "${DIST_OUT}"
python ../../paconvert/main.py -i . -o "${DIST_OUT}" --log_level "DEBUG"

echo '******************************************************************************'
echo "Running Distribute Unit Tests"
set +e

check_errors=0
failed_tests=()
test_list=$(ls *.py | grep -v run_and_compare.py)
for item in $test_list; do
    cmd1="torchrun --nproc_per_node=2 ${item}"
    cmd2="python -m paddle.distributed.launch ${DIST_OUT}/${item}"
    python run_and_compare.py "$cmd1" "$cmd2"
    if [ $? -ne 0 ]; then
        failed_tests+=("${item}")
        check_errors=1
    fi
done

echo '******************************************************************************'
if [ ${#failed_tests[@]} -ne 0 ]; then
    printf '%s\n' "${failed_tests[@]}" > failed_tests.txt
    echo "Your PR code Distributed unit test check FAILED"
    echo "The following distributed tests failed:"
    cat failed_tests.txt
    echo "Please run the following command:"
    echo ""
    echo "    cd tests/distributed && bash unittest_check_distribute.sh"
    echo ""
    echo "For more information, please refer to our check guides:"
    echo "https://github.com/paddlepaddle/paconvert#readme"
else
    echo "All tests PASSED!"
fi
echo '******************************************************************************'

exit ${check_errors}
