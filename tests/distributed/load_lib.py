import ctypes
import os

# 获取当前脚本文件的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建完整的库文件路径
cusparse_lib_path = os.path.join(script_dir, 'libcusparse.so.12')

# 加载动态链接库
libcusparse = ctypes.CDLL(cusparse_lib_path)

# 设置函数参数类型和返回类型
libcusparse.OnInit.argtypes = [ctypes.c_int]
libcusparse.OnInit.restype = ctypes.c_int

# 调用函数
result = libcusparse.OnInit(132)