from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from torch import randn
    from torch import matmul
    import warnings

if TYPE_CHECKING:
    import torch,numpy
    

if TYPE_CHECKING:
    a=randn(10, 20)
    b=randn(10, 20)
    c=matmul(a, b)
    d=torch.add(a,b)
    numpy.array([1,2,3])
else:
    warnings.warn("The above code is meaningless, in order to pass the code style CI")
