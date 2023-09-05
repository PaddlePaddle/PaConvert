from torch.nn import Module
from torch import Tensor
from torch import float32

class A(Module):
    def __init__(self, data: Tensor):
        pass


def func1(data: Tensor):
    data(x)
    data['id']
    func(data)

def func2() -> Tensor:
    pass

def func3(dtype=float32):
    pass


isinstance(x, Tensor)

setattr(Tensor, 'add', add_func)
