import torch
from transformers import AddedToken, GenerationConfig
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from flash_attn.flash_attn_interface import flash_attn_func

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
    ):
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
    output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(bsz, q_len, -1)

print('#########################case7#########################')
if attention_mask is None:
    output = flash_attn_func(q, k, v, 0.0, causal=True).view(bsz, q_len, -1)
