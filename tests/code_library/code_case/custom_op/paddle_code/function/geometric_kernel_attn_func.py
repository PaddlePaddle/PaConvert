import paddle
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def get_sources(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(current_dir, 'src')
    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, '*.cu'))
    sources = main_file
    sources += source_cuda
    sources = [os.path.join(extensions_dir, s) for s in sources]
    return extensions_dir, sources


custom_op_module = paddle.utils.cpp_extension.load(name=
    'GeometricKernelAttention', sources=get_sources()[1], extra_cxx_cflags=
    [], extra_cuda_cflags=['-DCUDA_HAS_FP16=1',
    '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__'], extra_include_paths=get_sources()[0])
