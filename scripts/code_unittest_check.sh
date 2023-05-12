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
/root/anaconda3/bin/pytest /workspace/$1/PaConvert/tests;check_error=$?

echo -e '************************************************************************************\n'
if [ ${check_error} != 0 ];then
    echo '************************************************************************************' > /dev/null
    echo "______                                   _   "  > /dev/null
    echo "| ___ \                                 | |  "  > /dev/null
    echo "| |_/ /_ _  ___ ___  _ ____   _____ _ __| |_ "  > /dev/null
    echo "|  __/ _  |/ __/ _ \\| \_ \ \ / / _ \ \__| __|"  > /dev/null
    echo "| | | (_| | (_| (_) | | | \\ V /  __/ |  | |_ "  > /dev/null
    echo "\\_|  \\__,_|\\___\\___/|_| |_|\\_/ \\___|_|   \\__|"  > /dev/null
    echo '************************************************************************************' > /dev/null
    echo "Your PR code unit test check failed." > /dev/null
    echo "Please run the following command." > /dev/null
    echo "" > /dev/null
    echo "    pytest tests" > /dev/null
    echo "" > /dev/null
    echo "For more information, please refer to our check guide:" > /dev/null
    echo "https://github.com/PaddlePaddle/PaConvert#readme." > /dev/null
else
    echo '************************************************************************************' > /dev/null
    echo "______                                   _   "  > /dev/null
    echo "| ___ \                                 | |  "  > /dev/null
    echo "| |_/ /_ _  ___ ___  _ ____   _____ _ __| |_ "  > /dev/null
    echo "|  __/ _  |/ __/ _ \\| \_ \ \ / / _ \ \__| __|"  > /dev/null
    echo "| | | (_| | (_| (_) | | | \\ V /  __/ |  | |_ "  > /dev/null
    echo "\\_|  \\__,_|\\___\\___/|_| |_|\\_/ \\___|_|   \\__|"  > /dev/null
    echo '************************************************************************************' > /dev/null
    echo "Your PR code unit test check passed."
    echo '************************************************************************************'
fi
exit ${check_error}

                                             
                                             
