import paddle
import paddlenlp
print('#########################case1#########################')


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
logger = paddle.utils.try_import("logging").getLogger(name=__name__)
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
    assert None == None or None == paddle.utils.try_import("math").sqrt(
        q.shape[-1]
    ), "Fault: The softmax_scale parameter defaults to the square root of the last dimension of query, not allowed manually set"
    assert (
        paddle.device.cuda.get_device_capability()[0] >= 8
    ), "Fault: Your device computational capabilities less 8"
    output = paddle.nn.functional.flash_attention.flash_attention(
        query=q, key=k, value=v, dropout=0.0, causal=True
    )[0]
print("#########################case7#########################")
if attention_mask is None:
    assert (
        paddle.device.cuda.get_device_capability()[0] >= 8
    ), "Fault: Your device computational capabilities less 8"
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
), "Fault: Your device computational capabilities less 8"
output = paddle.nn.functional.flash_attention.flash_attn_unpadded(
    query=q,
    key=k,
    value=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=seqlen_q,
    max_seqlen_k=seqlen_k,
    dropout=dropout_p,
    scale=paddle.utils.try_import("math").sqrt(q.shape[-1]),
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


print('#########################case16#########################')
utils.apply_rotary_position_embeddings(x=x, cos=cos, sin=sin)
print('#########################case17#########################')
paddle.incubate.nn.functional.fused_rms_norm(x, weight, paddle.zeros_like(
    weight), eps, len(x.shape) - 1)[0]
