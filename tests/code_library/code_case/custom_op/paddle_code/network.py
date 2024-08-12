import paddle
import GeometricKernelAttention
from .function import custom_op_module


class GeometryKernelAttention(paddle.nn.Layer):

    def forward(self, query, key=None, value=None, identity=None, query_pos
        =None, key_padding_mask=None, reference_points=None, spatial_shapes
        =None, level_start_index=None, **kwargs):
        """C++ Custom OP, only convert Python part, C++ part not support to convert, you must write C++/CUDA file manually"""
>>>        output1 = 
        GeometricKernelAttention.custom_op_xxx(value, spatial_shapes,
            level_start_index, sampling_locations, attention_weights, self.
            im2col_step)
        """C++ Custom OP, only convert Python part, C++ part not support to convert, you must write C++/CUDA file manually"""
>>>        output2 = 
        custom_op_module(value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, self.im2col_step)
        return output1, output2
