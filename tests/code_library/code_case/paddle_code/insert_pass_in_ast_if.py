from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    pass
    import warnings
if TYPE_CHECKING:
    import numpy
try:
    pass
except:
    flash_attn_qkvpacked_func = None
if True:
    pass
else:
    flash_attn_qkvpacked_func = None
if TYPE_CHECKING:
    a = paddle.randn(shape=[10, 20])
    b = paddle.randn(shape=[10, 20])
    c = paddle.matmul(a, b)
    d = paddle.add(a, b)
    numpy.array([1, 2, 3])
else:
    warnings.warn("The above code is meaningless, in order to pass the code style CI")
