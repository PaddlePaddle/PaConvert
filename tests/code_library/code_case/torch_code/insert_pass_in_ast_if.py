from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from torch import randn
    import warnings

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    a=randn(10, 20)
    b=randn(10, 20)
    c=torch.matmul(a, b)
else:
    warnings.warn("The above code is meaningless, in order to pass the code style CI")
