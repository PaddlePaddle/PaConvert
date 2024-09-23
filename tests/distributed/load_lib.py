# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import ctypes
import os


def load_cusparse_library():
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建完整的库文件路径
    cusparse_lib_path = os.path.join(script_dir, "libcusparse.so.12")

    # 检查库文件是否存在
    if not os.path.exists(cusparse_lib_path):
        raise FileNotFoundError(f"cuSPARSE library not found at {cusparse_lib_path}")

    # 尝试加载动态链接库
    try:
        libcusparse = ctypes.CDLL(cusparse_lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load cuSPARSE library: {e}")

    # 设置函数参数类型和返回类型
    libcusparse.OnInit.argtypes = [ctypes.c_int]
    libcusparse.OnInit.restype = ctypes.c_int

    return libcusparse


def main():
    try:
        libcusparse = load_cusparse_library()
        # 调用函数
        result = libcusparse.OnInit(132)
        # print("调用结果:", result)
    except Exception as e:
        # print(f"Error occurred: {e}")
        pass


if __name__ == "__main__":
    main()
