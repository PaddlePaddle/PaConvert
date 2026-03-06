import logging
from typing import Optional

import paddle
import paddleformers

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
setattr(paddleformers.transformers.model_utils.PretrainedModel, "get_head_mask", _get_head_mask)

original_generate = paddleformers.generation.utils.GenerationMixin.generate
def _generate(self, input_ids, *args, **kwargs):
    return paddle.concat((input_ids, original_generate(self,input_ids, *args, **kwargs)[0]), axis=-1)
setattr(paddleformers.generation.utils.GenerationMixin, "generate", _generate)

setattr(paddleformers.transformers.model_utils.PretrainedModel, "device", None)

def _post_init(self):
    if hasattr(self, "init_weights"):
        self.init_weights()
    elif hasattr(self, "_init_weights"):
        self._init_weights()
setattr(paddleformers.transformers.model_utils.PretrainedModel, "post_init", _post_init)

def apply_rotary_position_embeddings(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
):
    if seqlen_offsets not in (0, None):
        raise NotImplementedError(
            "PaConvert only supports apply_rotary_emb_func with default seqlen_offsets"
        )
    if cu_seqlens is not None or max_seqlen is not None:
        raise NotImplementedError(
            "PaConvert only supports apply_rotary_emb_func without cu_seqlens or max_seqlen"
        )
    if not isinstance(cos, paddle.Tensor):
        cos = paddle.to_tensor(
            cos, dtype=x.dtype, place=x.place, stop_gradient=True
        )
    if not isinstance(sin, paddle.Tensor):
        sin = paddle.to_tensor(
            sin, dtype=x.dtype, place=x.place, stop_gradient=True
        )

    def _rotate_half(x):
        if interleaved:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return paddle.reshape(
                paddle.stack((-x2, x1), axis=-1), shape=x.shape
            )
        x1, x2 = paddle.split(x, num_or_sections=2, axis=-1)
        return paddle.concat((-x2, x1), axis=-1)

    if interleaved:
        cos = paddle.repeat_interleave(cos, repeats=2, axis=-1)
        sin = paddle.repeat_interleave(sin, repeats=2, axis=-1)
    else:
        cos = paddle.concat([cos, cos], axis=-1)
        sin = paddle.concat([sin, sin], axis=-1)

    cos = cos.unsqueeze(axis=-2)
    sin = sin.unsqueeze(axis=-2)
    rotary_dim = cos.shape[-1]
    assert rotary_dim <= x.shape[-1]
    t_rot, t_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    out = paddle.concat(
        x=((t_rot * cos) + (_rotate_half(t_rot) * sin), t_pass), axis=-1
    )
    if inplace:
        paddle.assign(out, output=x)
        return x
    return out

def paddle_flash_attn_rms_norm(x, weight, epsilon):
    if weight is not None and x.place.is_gpu_place():
        try:
            out = paddle.incubate.nn.functional.fused_rms_norm(
                x, weight, paddle.zeros_like(weight), epsilon, len(x.shape) - 1
            )
            if isinstance(out, (tuple, list)):
                return out[0]
            return out
        except Exception:
            pass

    original_dtype = x.dtype
    if x.dtype in [paddle.float16, paddle.bfloat16]:
        compute_x = paddle.cast(x, "float32")
    else:
        compute_x = x

    out = compute_x * paddle.rsqrt(
        paddle.mean(paddle.square(compute_x), axis=-1, keepdim=True) + epsilon
    )
    if weight is not None:
        if weight.dtype != out.dtype:
            weight = paddle.cast(weight, out.dtype)
        out = out * weight

    if out.dtype != original_dtype:
        out = paddle.cast(out, original_dtype)
    return out
############################## 相关utils函数，如上 ##############################


print("#########################case1#########################")


def _add_tokens(
    self, new_tokens: Union[List[str], List[paddleformers.transformers.AddedToken]]
) -> int:
    return paddle.zeros([2, 4])


print("#########################case2#########################")


def chat(
    self,
    system: str = "You are a helpful assistant.",
    generation_config: Optional[paddleformers.generation.GenerationConfig] = None,
    logits_processor: Optional[paddleformers.generation.LogitsProcessorList] = None,
    stopping_criteria: Optional[paddleformers.generation.StoppingCriteriaList] = None,
) -> Union[paddleformers.transformers.model_outputs.BaseModelOutput, paddle.Tensor]:
    return paddle.zeros([2, 4])


print("#########################case3#########################")
logger = logging.getLogger(name=__name__)
print("#########################case4#########################")
paddleformers.transformers.model_outputs.BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=presents,
    hidden_states=all_hidden_states,
    attentions=all_self_attentions,
)
print("#########################case5#########################")
paddleformers.transformers.model_outputs.CausalLMOutputWithPast(
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
paddle.__version__.split(".")
print("#########################case9#########################")
paddle.__version__.split(".")
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


class StopWordsLogitsProcessor(paddleformers.generation.LogitsProcessor):
    pass


print("#########################case14#########################")


class QWenPreTrainedModel(paddleformers.transformers.PretrainedModel):
    pass


paddleformers.transformers.PretrainedModel(config)
paddleformers.transformers.PretrainedModel(config=config)
paddleformers.transformers.PretrainedModel(config, key1=value1, key2=value2)
print("#########################case15#########################")


class QWenTokenizer(paddleformers.PreTrainedTokenizer):
    pass


print("#########################case16#########################")
apply_rotary_position_embeddings(x=x, cos=cos, sin=sin)
print("#########################case17#########################")
paddle_flash_attn_rms_norm(x, weight, eps)
