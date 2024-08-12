import torch
from .function import GeometricKernelAttentionFunc, GeometricKernelAttentionFuncLoad

class GeometryKernelAttention(torch.nn.Module):
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        output1 = GeometricKernelAttentionFunc.apply(
            value, spatial_shapes, level_start_index, sampling_locations.contiguous(), attention_weights, self.im2col_step
        )
        output2 = GeometricKernelAttentionFuncLoad.apply(
            value, spatial_shapes, level_start_index, sampling_locations.contiguous(), attention_weights, self.im2col_step
        )
        return output1, output2
