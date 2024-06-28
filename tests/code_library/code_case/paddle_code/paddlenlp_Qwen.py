import sys
sys.path.append('tests/code_library/code_case/convert_paddle_code/utils')
import paddle_aux
import paddle
import paddlenlp
print('#########################case1#########################')


def _add_tokens(self, new_tokens: Union[List[str], List[paddlenlp.
    transformers.AddedToken]]) ->int:
    return paddle.zeros(shape=[2, 4])


print('#########################case2#########################')


def chat(self, system: str='You are a helpful assistant.',
    generation_config: Optional[paddlenlp.generation.GenerationConfig]=None,
    logits_processor: Optional[paddlenlp.generation.LogitsProcessorList]=
    None, stopping_criteria: Optional[paddlenlp.generation.
    StoppingCriteriaList]=None) ->Union[paddlenlp.transformers.
    model_outputs.BaseModelOutput, paddle.Tensor]:
    return paddle.zeros(shape=[2, 4])


print('#########################case3#########################')
logger = paddle.utils.try_import('logging').getLogger(name=__name__)
print('#########################case4#########################')
paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(last_hidden_state
    =hidden_states, past_key_values=presents, hidden_states=
    all_hidden_states, attentions=all_self_attentions)
print('#########################case5#########################')
paddlenlp.transformers.model_outputs.CausalLMOutputWithPast(loss=loss,
    logits=lm_logits, past_key_values=transformer_outputs.past_key_values,
    hidden_states=transformer_outputs.hidden_states, attentions=
    transformer_outputs.attentions)
print('#########################case6#########################')
if attention_mask is None:
    assert None is None or None is paddle.utils.try_import('math').sqrt(q.
        shape[-1]), 'Fault: Not support parameter softmax_scale'
    assert paddle.device.cuda.get_device_capability()[0
        ] >= 8, 'Fault: Your device computational capabilities less 8'
    output = paddle.nn.functional.flash_attention.flash_attention(query=q,
        key=k, value=v, dropout_p=0.0, causal=True)
print('#########################case7#########################')
if attention_mask is None:
    assert paddle.device.cuda.get_device_capability()[0
        ] >= 8, 'Fault: Your device computational capabilities less 8'
    output = paddle.nn.functional.flash_attention.flash_attention(query=q,
        key=k, value=v, dropout_p=0.0, causal=True)
print('#########################case8#########################')
paddle.__version__.split(sep='.')
print('#########################case9#########################')
paddle.__version__.split(sep='.')
print('#########################case10########################')
output = paddle.nn.functional.flash_attention.flash_attn_unpadded(query=q,
    key=k, value=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k, dropout_p=dropout_p,
    scale=self.softmax_scale, causal=is_causal)
print('#########################case11#########################')
paddle_aux.apply_rotary_emb_func(x=x, cos=cos, sin=sin)
print('#########################case12#########################')
paddle_aux.rms_norm(x=x, weight=weight, epsilon=eps)
print('#########################case13#########################')
outputs = paddle.distributed.fleet.utils.recompute(block, args1, args2, args3)
print('#########################case14#########################')


class QWenPreTrainedModel(paddlenlp.transformers.PretrainedModel):
    pass


print('#########################case15#########################')


class QWenTokenizer(paddlenlp.transformers.PretrainedTokenizer):
    pass


print('#########################case16#########################')


class QWenConfig(paddlenlp.transformers.PretrainedConfig):
    pass


print('#########################case18#########################')


class StopWordsLogitsProcessor(paddlenlp.generation.LogitsProcessor):
    pass
