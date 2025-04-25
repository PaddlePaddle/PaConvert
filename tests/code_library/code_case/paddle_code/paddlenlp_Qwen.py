import logging
from typing import Optional

import paddle
import paddlenlp

############################## 相关utils函数，如下 ##############################

def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
    return head_mask

def _get_head_mask(
    self,
    head_mask: Optional[paddle.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask
setattr(paddlenlp.transformers.model_utils.PretrainedModel, "get_head_mask", _get_head_mask)

original_generate = paddlenlp.generation.utils.GenerationMixin.generate
def _generate(self, input_ids, *args, **kwargs):
    return paddle.concat((input_ids, original_generate(self,input_ids, *args, **kwargs)[0]), axis=-1)
setattr(paddlenlp.generation.utils.GenerationMixin, "generate", _generate)

setattr(paddlenlp.transformers.model_utils.PretrainedModel, "device", None)

def _post_init(self):
    if hasattr(self, "init_weights"):
        self.init_weights()
    elif hasattr(self, "_init_weights"):
        self._init_weights()
setattr(paddlenlp.transformers.model_utils.PretrainedModel, "post_init", _post_init)

import paddlenlp

original_encode = paddlenlp.transformers.tokenizer_utils_base.PretrainedTokenizerBase.encode
def _encode(self, *args, **kwargs):
    return original_encode(self, *args, **kwargs)["input_ids"]
setattr(paddlenlp.transformers.tokenizer_utils_base.PretrainedTokenizerBase, "encode", _encode)

def apply_rotary_position_embeddings(x, cos, sin):
    if not isinstance(cos, paddle.Tensor):
        cos = paddle.to_tensor(cos)
    if not isinstance(sin, paddle.Tensor):
        sin = paddle.to_tensor(sin)

    def _rotate_half(x):
        from einops import rearrange

        x = rearrange(x, "... (j d) -> ... j d", j=2)
        x1, x2 = x.unbind(axis=-2)
        return paddle.concat((-x2, x1), axis=-1)
    # [seq_len,rotary_dim/2] ==>[seq_len, rotary_dim]
    cos = paddle.concat([cos,cos],axis=-1)
    # [seq_len, rotary_dim] ==>[1,seq_len, 1,rotary_dim]
    cos=cos.unsqueeze(axis=1).unsqueeze(axis=0)
    # [seq_len,rotary_dim/2] ==>[seq_len, rotary_dim]
    sin = paddle.concat([sin,sin],axis=-1)
    # [seq_len, rotary_dim] ==>[1,seq_len, 1,rotary_dim]
    sin=sin.unsqueeze(axis=1).unsqueeze(axis=0)
    t_rot, t_pass = x[..., :cos.shape[-1]], x[..., cos.shape[-1]:]
    t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)

    return paddle.concat(x=(t_rot, t_pass), axis=-1)
############################## 相关utils函数，如上 ##############################


print("#########################case1#########################")


def _add_tokens(
    self, new_tokens: Union[List[str], List[paddlenlp.transformers.AddedToken]]
) -> int:
    return paddle.zeros(shape=[2, 4])


print("#########################case2#########################")


def chat(
    self,
    system: str = "You are a helpful assistant.",
    generation_config: Optional[paddlenlp.generation.GenerationConfig] = None,
    logits_processor: Optional[paddlenlp.generation.LogitsProcessorList] = None,
    stopping_criteria: Optional[paddlenlp.generation.StoppingCriteriaList] = None,
) -> Union[paddlenlp.transformers.model_outputs.BaseModelOutput, paddle.Tensor]:
    return paddle.zeros(shape=[2, 4])


print("#########################case3#########################")
logger = logging.getLogger(name=__name__)
print("#########################case4#########################")
paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=presents,
    hidden_states=all_hidden_states,
    attentions=all_self_attentions,
)
print("#########################case5#########################")
paddlenlp.transformers.model_outputs.CausalLMOutputWithPast(
    loss=loss,
    logits=lm_logits,
    past_key_values=transformer_outputs.past_key_values,
    hidden_states=transformer_outputs.hidden_states,
    attentions=transformer_outputs.attentions,
)
print("#########################case6#########################")
if attention_mask is None:
    output = paddle.nn.functional.flash_attention.flash_attention(
        query=q, key=k, value=v, dropout=0.0, causal=True
    )[0]
print("#########################case7#########################")
if attention_mask is None:
    output = paddle.nn.functional.flash_attention.flash_attention(
        query=q, key=k, value=v, dropout=0.0, causal=True
    )[0]
print("#########################case8#########################")
paddle.__version__.split(sep=".")
print("#########################case9#########################")
paddle.__version__.split(sep=".")
print("#########################case10########################")
assert (
    paddle.device.cuda.get_device_capability()[0] >= 8
), "Device capabilities should be at least 8"
output = paddle.nn.functional.flash_attention.flash_attn_unpadded(
    query=q,
    key=k,
    value=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=seqlen_q,
    max_seqlen_k=seqlen_k,
    dropout=dropout_p,
    scale=self.softmax_scale,
    causal=is_causal,
)[0]
print("#########################case11#########################")
outputs = paddle.distributed.fleet.utils.recompute(block, args1, args2, args3)
print("#########################case12#########################")


class QWenConfig(PretrainedConfig):
    pass


print("#########################case13#########################")


class StopWordsLogitsProcessor(paddlenlp.generation.LogitsProcessor):
    pass


print("#########################case14#########################")


class QWenPreTrainedModel(paddlenlp.transformers.PretrainedModel):
    pass


paddlenlp.transformers.PretrainedModel(config)
paddlenlp.transformers.PretrainedModel(config=config)
paddlenlp.transformers.PretrainedModel(config, key1=value1, key2=value2)
print("#########################case15#########################")


class QWenTokenizer(paddlenlp.transformers.PretrainedTokenizer):
    pass


print("#########################case16#########################")
apply_rotary_position_embeddings(x=x, cos=cos, sin=sin)
print("#########################case17#########################")
paddle.incubate.nn.functional.fused_rms_norm(
    x, weight, paddle.zeros_like(weight), eps, len(x.shape) - 1
)[0]
