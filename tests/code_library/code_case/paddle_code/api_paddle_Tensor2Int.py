import paddle
print('#########################case1#########################')


def a(x: paddle.int32):
    pass


print('#########################case2#########################')
a = paddle.empty(shape=[2, 3, 6], dtype='int32')
