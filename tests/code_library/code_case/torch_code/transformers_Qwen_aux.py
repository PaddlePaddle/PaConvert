from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from flash_attn.layers.rotary import apply_rotary_emb_func
from flash_attn.ops.rms_norm import rms_norm


print('#########################case1#########################')
class QWenPreTrainedModel(PreTrainedModel):
    pass

print('#########################case2#########################')
class QWenTokenizer(PreTrainedTokenizer):
    pass

print('#########################case3#########################')
apply_rotary_emb_func(x, cos, sin)

print('#########################case4#########################')
rms_norm(x, weight, eps)
