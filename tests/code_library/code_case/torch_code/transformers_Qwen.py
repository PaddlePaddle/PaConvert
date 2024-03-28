import torch
from transformers import AddedToken, GenerationConfig
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
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
