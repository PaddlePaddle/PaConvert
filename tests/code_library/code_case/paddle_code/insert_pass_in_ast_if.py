from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    import warnings
if TYPE_CHECKING:
    import numpy
if TYPE_CHECKING:
    a = paddle.randn(shape=[10, 20])
    b = paddle.randn(shape=[10, 20])
    c = paddle.matmul(x=a, y=b)
    d = paddle.add(x=a, y=paddle.to_tensor(b))
    numpy.array([1, 2, 3])
else:
    warnings.warn("The above code is meaningless, in order to pass the code style CI")
