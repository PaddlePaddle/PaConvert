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

# pre-commit multi-thread running.
echo "Checking code unit test by pytest ..."
pytest tests;check_error=$?
pwd
echo -e '\n************************************************************************************'
if [ ${check_error} != 0 ];then
    echo "Your PR code unit test check failed."
    echo "Please run the following command"
    echo ""
    echo "    pytest tests"
    echo ""
    echo "For more information, please refer to our check guide:"
    echo "https://github.com/PaddlePaddle/PaConvert#readme"
else
    echo "Your PR code unit test check passed."
fi
echo -e '************************************************************************************\n'

exit ${check_error}

                                             
                                             
