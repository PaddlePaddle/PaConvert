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

TORCH_PROJECT_PATH="${TORCH_PROJECT_PATH:-/workspace/torch_project}"

echo '******************************************************************************'
echo "Installing develop CPU version paddle"
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu || true
python -m pip install --force-reinstall --no-cache-dir -U --pre paddlepaddle \
    -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 120 --retries 3
python -c "import paddle; print('paddle version: ', paddle.__version__); print('paddle commit info: ', paddle.__git_commit__)"

echo '******************************************************************************'
echo "Installing paconvert requirements"
python -m pip install -r requirements.txt
python -m pip install pandas openpyxl || true

set +e

echo '******************************************************************************'
echo "[code-set-convert] Start converting code set under ${TORCH_PROJECT_PATH}"
if [ ! -d "${TORCH_PROJECT_PATH}" ]; then
    echo "${TORCH_PROJECT_PATH} is not a valid directory. Please stage the model code set on the runner host."
    exit 1
fi

shopt -s nullglob
projects=("${TORCH_PROJECT_PATH}"/*)
if [ ${#projects[@]} -eq 0 ]; then
    echo "${TORCH_PROJECT_PATH} is empty. Please stage the model code set on the runner host."
    exit 1
fi

failed_project=()
for project in "${projects[@]}"; do
    if [ -d "$project" ]; then
        project_name=$(basename "$project")
        echo "[code-set-convert] Converting project: $project_name"
        if ! python paconvert/main.py --in_dir "$project" --show_unsupport_api --calculate_speed; then
            failed_project+=("$project_name")
        fi
    fi
done

if [ ${#failed_project[@]} -ne 0 ]; then
    printf '%s\n' "${failed_project[@]}" > failed_projects.txt
    echo "[code-set-convert] The following projects fail to convert:"
    cat failed_projects.txt
    exit 1
fi

echo '******************************************************************************'
echo "[modeltest] Start modeltest"
python tools/modeltest/modeltest_check.py
check_errors=$?

echo '******************************************************************************'
if [ ${check_errors} -ne 0 ]; then
    echo "Your PR code modeltest check FAILED"
else
    echo "All Modeltest PASSED!"
fi
echo '******************************************************************************'

exit ${check_errors}
