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

echo '******************************************************************************'
echo "Installing develop GPU version paddle"
# python3.10 -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true
# python3.10 -m pip install --force-reinstall --no-cache-dir --no-deps -U --pre paddlepaddle-gpu \
#     -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/ \
#     --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
#     --timeout 120 --retries 3
python3.10 -m pip install safetensors==0.6.2
python3.10 -c "import paddle; print('paddle version: ', paddle.__version__); print('paddle commit info: ', paddle.__git_commit__)"

echo '******************************************************************************'
echo "Installing paconvert requirements"
python3.10 -m pip install -r requirements.txt
if [ -f tests/requirements.txt ]; then
    python3.10 -m pip install -r tests/requirements.txt
fi

echo '******************************************************************************'
python3.10 -c "import torch; print('torch version: ', torch.__version__, '| cuda available: ', torch.cuda.is_available())"

echo '******************************************************************************'
echo "Checking code gpu unit test by pytest ..."
set +e

PYTEST_IGNORE=(
    --ignore=tests/test_hub_download_url_to_file.py
    --ignore=tests/test_hub_help.py
    --ignore=tests/test_hub_list.py
    --ignore=tests/test_hub_load.py
    --ignore=tests/test_hub_load_state_dict_from_url.py
)

# Run test_cuda_stream.py separately and FIRST (GPU state is clean),
# as it can segfault when run after other GPU tests (Paddle FullKernel issue).
# Running in isolation prevents the segfault from killing the entire test batch.
python3.10 -m pytest -v -s -p no:warnings tests/test_cuda_stream.py 2>&1 | tee pytest.log
stream_exit=${PIPESTATUS[0]}

# Run test_cuda_CUDAGraph.py in its own process as well: CUDA graph
# capture/replay leaves Paddle GPU state that can natively crash later
# AMP tests (e.g. test_cuda_amp_GradScaler) in the same pytest worker.
python3.10 -m pytest -v -s -p no:warnings tests/test_cuda_CUDAGraph.py 2>&1 | tee -a pytest.log
cudagraph_exit=${PIPESTATUS[0]}

# set_stream installs a process-global Paddle stream owned by the temporary exec
# namespace. Run it separately so namespace cleanup cannot poison later GPU tests.
python3.10 -m pytest -v -s -p no:warnings tests/test_cuda_set_stream.py 2>&1 | tee -a pytest.log
setstream_exit=${PIPESTATUS[0]}

python3.10 -m pytest -v -s -p no:warnings "${PYTEST_IGNORE[@]}" \
    --ignore=tests/test_cuda_stream.py \
    --ignore=tests/test_cuda_CUDAGraph.py \
    --ignore=tests/test_cuda_set_stream.py \
    -n 1 --reruns=3 ./tests 2>&1 | tee -a pytest.log
check_errors=${PIPESTATUS[0]}
if [ ${check_errors} -ne 0 ]; then
    echo "Rerun GPU unit test"
    python3.10 -m pytest -v -s -p no:warnings "${PYTEST_IGNORE[@]}" \
        --ignore=tests/test_cuda_stream.py \
        --ignore=tests/test_cuda_CUDAGraph.py \
        --ignore=tests/test_cuda_set_stream.py \
        -n 1 --lf ./tests 2>&1 | tee -a pytest.log
    check_errors=${PIPESTATUS[0]}
fi

# Propagate isolated test failures if any
if [ ${stream_exit} -ne 0 ]; then
    check_errors=${stream_exit}
fi
if [ ${cudagraph_exit} -ne 0 ]; then
    check_errors=${cudagraph_exit}
fi
if [ ${setstream_exit} -ne 0 ]; then
    check_errors=${setstream_exit}
fi

echo '******************************************************************************'
if [ ${check_errors} -ne 0 ]; then
    echo "Your PR code GPU unit test check FAILED"
    echo "Please run the following command:"
    echo ""
    echo "    pytest -m pytest tests"
    echo ""
    echo "For more information, please refer to our check guides:"
    echo "https://github.com/paddlepaddle/paconvert#readme"
else
    echo "All tests PASSED!"
fi
echo '******************************************************************************'

exit ${check_errors}
