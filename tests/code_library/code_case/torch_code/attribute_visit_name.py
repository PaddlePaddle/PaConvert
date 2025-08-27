rom torch.nn import Module
from torch import Tensor
from torch import float32
from torch import nn

from transformers.generation.utils import GenerateOutput
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import torch.utils.data as data # noqa: F401

import torch.add as TorchAdd # noqa: F401

class A(Module):
    def __init__(self, data: Tensor):
        pass


def func1(data: Tensor):
    data(x)
    data.data_loader(x)
    data['id']
    func(data)
    data = json.load(f)
    self.data = self.__make_dataset()

def func2(data) -> Tensor:
    pass

def func3(dtype=float32):
    pass


isinstance(x, Tensor)

setattr(Tensor, 'add', add_func)

hasattr(Tensor, 'add')

Union[GenerateOutput, torch.LongTensor]
Optional[Tensor] = None

my_add = TorchAdd

setattr(nn, 'functional', my_functional_module)

ACT2FN['tanh']
ALL_ATTENTION_FUNCTIONS['flash_attention_2']
ROPE_INIT_FUNCTIONS['linear']
