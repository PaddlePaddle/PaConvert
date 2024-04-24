import paddle
from typing import Optional, Tuple, Union, List
print('#########################case1#########################')
Union[Tuple, paddle.Tensor]
print('#########################case2#########################')
paddle.to_tensor(data=[True, False], dtype='bool')
print('#########################case3#########################')


def fun(b: paddle.Tensor):
    pass


print('#########################case4#########################')
paddle.set_default_dtype(d=paddle.float16)
print('#########################case5#########################')


def forward(hidden_states: Optional[Tuple[paddle.Tensor]],
    rotary_pos_emb_list: Optional[List[List[paddle.Tensor]]]=None,
    layer_past: Optional[Tuple[paddle.Tensor]]=None, attention_mask:
    Optional[paddle.Tensor]=None, head_mask: Optional[paddle.Tensor]=None,
    encoder_hidden_states: Optional[paddle.Tensor]=None,
    encoder_attention_mask: Optional[paddle.Tensor]=None, output_attentions:
    Optional[bool]=False, use_cache: Optional[bool]=False):
    pass
