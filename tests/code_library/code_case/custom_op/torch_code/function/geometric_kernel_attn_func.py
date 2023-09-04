from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.autograd import Function
from torch.autograd.function import once_differentiable

import GeometricKernelAttention as GKA


class GeometricKernelAttentionFunc(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = GKA.geometric_kernel_attn_cuda_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_attn_weight = \
            GKA.geometric_kernel_attn_cuda_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, None, grad_attn_weight, None


def get_sources(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(current_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    # source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = main_file
    sources += source_cuda
    sources = [os.path.join(extensions_dir, s) for s in sources]
    return extensions_dir, sources


from torch.utils.cpp_extension import load

custom_op_module = load(name='GeometricKernelAttention', sources=get_sources()[1], extra_cflags=[], extra_cuda_cflags=[
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ], extra_include_paths=get_sources()[0])


class GeometricKernelAttentionFuncLoad(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = custom_op_module.geometric_kernel_attn_cuda_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_attn_weight = \
            custom_op_module.geometric_kernel_attn_cuda_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, None, grad_attn_weight, None
