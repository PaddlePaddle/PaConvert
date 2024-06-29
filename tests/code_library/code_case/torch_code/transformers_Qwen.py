import torch
from transformers import AddedToken, GenerationConfig, StoppingCriteriaList
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.generation import LogitsProcessor
from transformers.generation.utils import GenerateOutput
from transformers.generation.logits_process import LogitsProcessorList
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_unpadded_func


print('#########################case1#########################')
def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
    ) -> int:
    return torch.zeros([2,4])
print('#########################case2#########################')
def chat(
        self,
        system: str = "You are a helpful assistant.",
        generation_config: Optional[GenerationConfig] = None,  
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> Union[GenerateOutput, torch.Tensor]:
    return torch.zeros([2,4])
print('#########################case3#########################')
logger = logging.get_logger(__name__)
print('#########################case4#########################')
BaseModelOutputWithPast(
                        last_hidden_state=hidden_states,
                        past_key_values=presents,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attentions,
                        )
print('#########################case5#########################')
CausalLMOutputWithPast(
                        loss=loss,
                        logits=lm_logits,
                        past_key_values=transformer_outputs.past_key_values,
                        hidden_states=transformer_outputs.hidden_states,
                        attentions=transformer_outputs.attentions,
                    )
print('#########################case6#########################')
if attention_mask is None:
    output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True)

print('#########################case7#########################')
if attention_mask is None:
    output = flash_attn_func(q, k, v, 0.0, causal=True)

print('#########################case8#########################')
torch.__version__.split(".")

print('#########################case9#########################')
flash_attn.__version__.split(".")

print('#########################case10########################')
output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )
print('#########################case11#########################')
outputs = torch.utils.checkpoint.checkpoint(block,args1,args2,args3)

print('#########################case12#########################')
class QWenConfig(PretrainedConfig):
    pass

print('#########################case13#########################')
class StopWordsLogitsProcessor(LogitsProcessor):
    pass
