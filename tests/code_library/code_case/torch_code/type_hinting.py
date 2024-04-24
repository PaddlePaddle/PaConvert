import torch
from typing import Optional, Tuple, Union, List

print('#########################case1#########################')
Union[Tuple,torch.BoolTensor]
print('#########################case2#########################')
torch.BoolTensor([True,False])
print('#########################case3#########################')
def fun(b: torch.BoolTensor):
    pass
print('#########################case4#########################')
torch.set_default_tensor_type(torch.cuda.HalfTensor)
print('#########################case5#########################')
def forward(
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
    pass
