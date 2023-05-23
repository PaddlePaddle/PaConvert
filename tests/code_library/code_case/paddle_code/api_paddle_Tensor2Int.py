import paddle
print('#########################case1#########################')


def a(x: paddle.Tensor):
    pass


print('#########################case2#########################')
a = paddle.empty(shape=[2, 3, 6], dtype='int32')
