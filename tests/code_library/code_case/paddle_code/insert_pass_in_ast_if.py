import paddle
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
    import warnings
if TYPE_CHECKING:
    pass
if TYPE_CHECKING:
    a = paddle.randn(shape=[10, 20])
    b = paddle.randn(shape=[10, 20])
    c = paddle.matmul(x=a, y=b)
else:
    warnings.warn(
        'The above code is meaningless, in order to pass the code style CI')
