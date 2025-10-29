from typing import List, Optional, Tuple, Union

import paddle

print("#########################case1#########################")
Union[Tuple, paddle.BoolTensor]
print("#########################case2#########################")
paddle.BoolTensor([True, False])
print("#########################case3#########################")


def fun(b: paddle.BoolTensor):
    pass


print("#########################case4#########################")
paddle.set_default_dtype(d=paddle.cuda.HalfTensor)
print("#########################case5#########################")


def forward(
    hidden_states: Optional[Tuple[paddle.FloatTensor]],
    rotary_pos_emb_list: Optional[List[List[paddle.Tensor]]] = None,
    layer_past: Optional[Tuple[paddle.Tensor]] = None,
    attention_mask: Optional[paddle.FloatTensor] = None,
    head_mask: Optional[paddle.FloatTensor] = None,
    encoder_hidden_states: Optional[paddle.Tensor] = None,
    encoder_attention_mask: Optional[paddle.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):
    pass
